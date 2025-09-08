# src/utils/rdf_utils.py
"""
Differentiable RDF utilities and atom count estimation.

Functions:
- differentiable_rdf(positions, bin_centers, sigma=0.1, lattice=None, device='cpu'):
    Compute a differentiable approximation of g(r) using Gaussian kernel histogram.
    positions: torch.Tensor, shape (N,3)
    bin_centers: torch.Tensor, shape (M,) - centers of r bins (Å)
    lattice: optional numpy array or torch tensor (3,3) lattice vectors in Å for PBC handling
    returns: torch.Tensor shape (M,)

- estimate_n_atoms_from_density(formula, density_gcm3, lattice_vector):
    Estimate integer number of atoms for the cell given density (g/cm^3),
    chemical formula string (e.g., "SiO2") and cell lattice vectors (3x3 in Å).
"""
import math
from typing import Optional, List, Tuple

import numpy as np
import torch

try:
    # ASE provides atomic_masses and atomic_numbers mapping
    from ase.data import atomic_masses, atomic_numbers
except Exception:
    atomic_masses = None
    atomic_numbers = None


def _parse_formula(formula: str) -> List[Tuple[str, int]]:
    import re
    token_re = re.compile(r'([A-Z][a-z]*)(\d*)')
    tokens = token_re.findall(formula)
    if not tokens:
        return []
    parsed = []
    for el, num in tokens:
        cnt = int(num) if num else 1
        parsed.append((el, cnt))
    return parsed


def estimate_molar_mass(formula: str) -> float:
    """
    Estimate molar mass (g/mol) of a chemical formula using ASE atomic_masses.
    """
    parsed = _parse_formula(formula)
    if not parsed:
        raise ValueError(f"Cannot parse formula: {formula}")
    if atomic_numbers is None or atomic_masses is None:
        raise RuntimeError("ASE atomic data not available. Install ase.")
    total = 0.0
    for el, cnt in parsed:
        Z = atomic_numbers.get(el, None)
        if Z is None:
            raise ValueError(f"Unknown element symbol: {el}")
        mass = atomic_masses[Z]  # atomic_masses indexed by atomic number
        total += mass * cnt
    return total


def estimate_n_atoms_from_density(formula: str, density_gcm3: float, lattice_vector: np.ndarray) -> int:
    """
    Estimate number of atoms in the given cell given density [g/cm^3].
    lattice_vector: (3,3) array in Å
    """
    # Cell volume in Å^3
    V_A3 = abs(np.linalg.det(np.array(lattice_vector, dtype=float)))
    # convert to cm^3
    V_cm3 = V_A3 * 1e-24
    # molar mass per formula unit (g/mol)
    molar_mass = estimate_molar_mass(formula)
    # Avogadro
    NA = 6.02214076e23
    # number of formula units in cell
    n_formula_units = (density_gcm3 * V_cm3 / molar_mass) * NA
    # atoms per formula unit
    parsed = _parse_formula(formula)
    atoms_per_unit = sum(cnt for _, cnt in parsed)
    n_atoms = int(round(n_formula_units * atoms_per_unit))
    if n_atoms < 1:
        n_atoms = max(1, int(atoms_per_unit))
    return n_atoms


def differentiable_rdf(
    positions: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float = 0.1,
    lattice: Optional[np.ndarray] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute a differentiable RDF g(r) using Gaussian kernels.

    Args:
        positions: (N,3) torch tensor (Å)
        bin_centers: (M,) torch tensor centers (Å)
        sigma: width of Gaussian kernel (Å)
        lattice: optional (3,3) numpy array (Å) for PBC (minimum image)
        device: torch device string

    Returns:
        g_r: (M,) torch tensor (same device)
    Notes:
        - Uses i<j pair counting (no double-counting)
        - Normalization follows: g(r) ~ (2 * sum_kernel) / (N * rho * 4π r^2 dr)
          with kernel integrated approximately by normalization factor sqrt(2π)*sigma.
    """
    if not torch.is_tensor(positions):
        positions = torch.tensor(positions, dtype=torch.float32, device=device)
    else:
        positions = positions.to(device=device, dtype=torch.float32)

    bin_centers = bin_centers.to(device=device, dtype=torch.float32)
    N = positions.shape[0]
    if N < 2:
        return torch.zeros_like(bin_centers, device=device)

    # Apply minimum image convention if lattice provided
    if lattice is not None:
        lattice = np.asarray(lattice, dtype=float)
        inv_lat = np.linalg.inv(lattice)
        pos_np = positions.detach().cpu().numpy()
        frac = (pos_np @ inv_lat.T)
        frac = frac - np.round(frac)  # wrap to [-0.5,0.5]
        pos_wrap = frac @ lattice.T
        pos = torch.tensor(pos_wrap, dtype=positions.dtype, device=device)
    else:
        pos = positions

    # pairwise distances (i<j)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N,N,3)
    dists = torch.norm(diff, dim=-1)  # (N,N)
    idx = torch.triu_indices(N, N, offset=1)
    pair_dists = dists[idx[0], idx[1]]  # (P,)

    # constants
    V_cell = 1.0
    if lattice is not None:
        V_cell = abs(np.linalg.det(np.array(lattice)))
    rho = N / V_cell  # atoms per Å^3

    r = bin_centers  # (M,)
    if r.shape[0] > 1:
        dr = float(r[1].item() - r[0].item())
    else:
        dr = 0.1

    norm_factor = math.sqrt(2.0 * math.pi) * sigma  # approx integral of Gaussian

    # kernel: shape (P, M)
    diff_r = pair_dists.unsqueeze(1) - r.unsqueeze(0)  # (P, M)
    kernel = torch.exp(-0.5 * (diff_r / sigma) ** 2) / norm_factor

    # sum over pairs (each pair i<j counted once) -> multiply by 2 for i!=j summation convention
    g_r_unnorm = kernel.sum(dim=0) * 2.0  # (M,)

    denom = (N * rho * (4.0 * math.pi * (r ** 2) * dr)).to(device)
    denom = torch.where(denom == 0, torch.ones_like(denom, device=device), denom)
    g_r = g_r_unnorm / denom
    return g_r


class StructureAnalysis:
    def __init__(self):
        self.structure = None
        self.cn_lim = [0, 10]
        self.distance_matrices = {}
        self.cache_dir = '__cache__'
        self.filename = None

    def load_structure(self, filename: str, file_format: str, **kwargs):
        """
        Load atomic structure from file.
        
        Args:
            filename (str): Path to the input file.
            file_format (str): Format of the input file ('vasp' or 'lammps-data').
            **kwargs: Additional keyword arguments for ase.io.read function.
        """
        # Reinitialize the distance_matrix each time the structure is reloaded
        self.distance_matrices.clear()
        self.filename = Path(filename).stem
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if file_format not in ['vasp', 'lammps-data','extxyz']:
            raise ValueError("Unsupported file format. Use 'vasp', 'lammps-data', or 'extxyz'.")
        default_args = {
            'vasp': {'index': None},
            'lammps-data': {'index': None,
                            'style': 'atomic'},
            'extxyz': {'index': None},
        }

        # Merge default arguments with user-provided kwargs
        read_args = {**default_args[file_format], **kwargs}
        try:
            self.structure = read(filename, format=file_format, **read_args)
        except Exception as e:
            raise IOError(f"Failed to load structure: {str(e)}")

    def calculate_distance_matrix(self, atoms: ase.atoms.Atoms, idx: int = None):
        if idx is None:
            idx = 0
        atoms_id = id(atoms)
        cache_file = f"{self.cache_dir}/{self.filename}_structure_{idx}.npy"
        if atoms_id in self.distance_matrices:
            return self.distance_matrices[atoms_id]

        global_dist = None
        if rank == 0:
            if os.path.exists(cache_file):
                global_dist = np.load(cache_file)
                print(f"Loaded from cache: {cache_file}")
        global_dist = comm.bcast(global_dist, root=0)
        if global_dist is not None:
            self.distance_matrices[atoms_id] = global_dist
            return global_dist
        
        nions = atoms.get_global_number_of_atoms()
        local_nions = nions // size
        start = rank * local_nions
        end = nions if rank == size - 1 else (rank + 1) * local_nions
        local_dist = np.zeros((end - start, nions))
        for i in range(start, end):
            local_dist[i - start, i:nions] = atoms.get_distances(i, range(i, nions), mic=True)
        local_size = local_dist.size
        sizes = comm.allgather(local_size)
        global_size = sum(sizes)
        displacements = [sum(sizes[:i]) for i in range(size)]
        global_dist = np.zeros(global_size).reshape([-1, local_dist.shape[1]])
        comm.Allgatherv(sendbuf=local_dist, recvbuf=(global_dist, sizes, displacements, MPI.DOUBLE))
        global_dist += global_dist.T - np.diag(np.diag(global_dist))
        np.fill_diagonal(global_dist, np.inf)
        
        if rank == 0:
            np.save(cache_file, global_dist)
            print(f"Saved to cache: {cache_file}")
        self.distance_matrices[atoms_id] = global_dist
        return global_dist
    
    def calculate_single_rdf(self, atoms: ase.atoms.Atoms, rmax: float, cutoff: float, dr: float, idx: int = None):
        distance_matrix = self.calculate_distance_matrix(atoms, idx)
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        rdf = np.zeros(len(bins) - 1)
        if rmax > atoms.get_cell().diagonal().min() / 2:
            print('WARNING: The input maximum radius is over the half the smallest cell dimension.')
        global_dist = distance_matrix
        nions = atoms.get_global_number_of_atoms()
        res, bin_edges = np.histogram(global_dist, bins=bins)
        rdf += res / ((nions ** 2 / atoms.get_volume()) * 4 * np.pi * dr * bin_edges[:-1] ** 2)
        coordination_numbers = np.sum(global_dist < cutoff, axis=1)
        return rdf, bin_edges, coordination_numbers

    def calculate_rdf(self, rmax, cutoff=2.0, dr=0.02):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        if isinstance(self.structure[0], ase.atom.Atom):
            rdf, bin_edges, coordination_numbers = self.calculate_single_rdf(self.structure, rmax, cutoff, dr)
        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            rdf = np.zeros(len(bins) - 1)
            for idx, atoms in enumerate(self.structure):
                if idx == 0: 
                    nions = atoms.get_global_number_of_atoms()
                    coordination_numbers = np.zeros(nimg*nions)
                single_rdf, bin_edges, single_coordination_numbers = self.calculate_single_rdf(atoms, rmax, cutoff, dr, idx=idx)
                rdf += single_rdf
                coordination_numbers[idx*nions: (idx+1)*nions] = single_coordination_numbers
            rdf /= nimg
        coordination_numbers = coordination_numbers.astype(int)
        min_cn, max_cn = self.cn_lim[0], self.cn_lim[1]
        cn_counts = np.bincount(coordination_numbers, minlength=max_cn - min_cn + 1)
        if np.max(coordination_numbers) > max_cn:
            print(f"WARNING: Some CN > {max_cn}, not included in distribution.")
        cn_labels = np.arange(min_cn, max_cn + 1)
        cn_sum = np.sum(cn_counts)
        return np.column_stack((bin_edges[:-1], rdf)), np.column_stack((cn_labels, cn_counts, cn_counts / cn_sum if cn_sum > 0 else cn_counts))

    def calculate_single_prdf(self, atoms: ase.atoms.Atoms, targets: tuple, rmax: float, cutoff: float, dr: float, idx: int = None):
        distance_matrix = self.calculate_distance_matrix(atoms, idx)
        (elemA, elemB) = targets
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        prdf = np.zeros(len(bins) - 1)
        if rmax > atoms.get_cell().diagonal().min() / 2:
            print('WARNING: The input maximum radius is over the half the smallest cell dimension.')
        sym = np.array(atoms.get_chemical_symbols())
        idA = np.where( sym == elemA )[0]
        nelemA = len(idA)
        idB = np.where( sym == elemB )[0]
        nelemB = len(idB)
        global_dist = distance_matrix[idA][:, idB]
        res, bin_edges = np.histogram(global_dist, bins=bins)
        prdf += res / (nelemA * nelemB / atoms.get_volume() * 4 * np.pi * dr * bin_edges[:-1] ** 2)
        coordination_numbers = np.sum(global_dist < cutoff, axis=1)
        return prdf, bin_edges, coordination_numbers

    def calculate_prdf(self, targets:tuple, rmax:float, cutoff:float=2.0, dr:float=0.02):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        if isinstance(self.structure[0], ase.atom.Atom):
            prdf, bin_edges, coordination_numbers = self.calculate_single_prdf(self.structure, targets, rmax, cutoff, dr)
        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            prdf = np.zeros(len(bins) - 1)
            for idx, atoms in enumerate(self.structure):
                if idx == 0: 
                    sym = np.array(atoms.get_chemical_symbols())
                    nions = len(np.where( sym == elemA )[0])
                    coordination_numbers = np.zeros(nimg*nions)
                single_prdf, bin_edges, single_coordination_numbers = self.calculate_single_prdf(atoms, targets, rmax, cutoff, dr, idx=idx)
                prdf += single_prdf
                coordination_numbers[idx*nions: (idx+1)*nions] = single_coordination_numbers
            prdf /= nimg
        coordination_numbers = coordination_numbers.astype(int)
        min_cn, max_cn = self.cn_lim[0], self.cn_lim[1]
        cn_counts = np.bincount(coordination_numbers, minlength=max_cn - min_cn + 1)
        if np.max(coordination_numbers) > max_cn:
            print(f"WARNING: Some CN > {max_cn}, not included in distribution.")
        cn_labels = np.arange(min_cn, max_cn + 1)
        cn_sum = np.sum(cn_counts)
        return np.column_stack((bin_edges[:-1], prdf)), np.column_stack((cn_labels, cn_counts, cn_counts / cn_sum if cn_sum > 0 else cn_counts))
