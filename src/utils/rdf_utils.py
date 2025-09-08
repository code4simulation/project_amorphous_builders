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

def differentiable_partial_rdf(
    atoms,
    elem1: str,
    elem2: str,
    rmax: float,
    step: float,
    sigma: float = 0.1,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Gaussian 커널 기반 미분 가능한 Partial RDF (elem1-elem2) 계산 (PBC/MIC 지원).

    Args:
        atoms: ASE Atoms 객체
        elem1: 첫 번째 원소명 (예: 'Si')
        elem2: 두 번째 원소명 (예: 'O')
        rmax: RDF 최대 거리 (Å)
        step: bin 간격 (Å)
        sigma: Gaussian kernel 폭 (Å)
        device: 계산 디바이스 ('cpu' 또는 'cuda')

    Returns:
        torch.Tensor: (bin_centers,) 미분 가능한 partial RDF 값
    """
    # positions, chemical symbols, cell/lattice 정보 추출
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.get_cell())
    N = positions.shape[0]
    pbc = getattr(atoms, 'pbc', [False, False, False])

    # 대상 원소 인덱스
    idx1 = np.where(symbols == elem1)[0]
    idx2 = np.where(symbols == elem2)[0]
    if len(idx1) == 0 or len(idx2) == 0:
        return torch.zeros(int(rmax/step), device=device)

    # bin 정의
    bin_edges = np.arange(0, rmax+step, step)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32, device=device)

    # positions tensor
    pos1 = torch.tensor(positions[idx1], dtype=torch.float32, device=device)  # (A,3)
    pos2 = torch.tensor(positions[idx2], dtype=torch.float32, device=device)  # (B,3)
    lattice = torch.tensor(cell, dtype=torch.float32, device=device)          # (3,3)
    inv_lat = torch.linalg.inv(lattice)                                       # (3,3)
    pbc_mask = torch.tensor(pbc, device=device, dtype=torch.bool)             # (3,)

    # 쌍 간 차이 벡터 (broadcast)
    dr = pos2.unsqueeze(0) - pos1.unsqueeze(1)  # (A,B,3)

    # 차이 벡터를 분수 좌표로 변환
    dfrac = torch.matmul(dr, inv_lat)  # (A,B,3)
    # MIC [-0.5,0.5) 래핑(PBC축만)
    if pbc_mask.any():
        dfrac_wrapped = dfrac.clone()
        for i in range(3):
            if pbc_mask[i]:
                dfrac_wrapped[..., i] -= torch.floor(dfrac_wrapped[..., i] + 0.5)
    else:
        dfrac_wrapped = dfrac
    # 다시 카테시안으로: 최소 영상 벡터
    dr_mic = torch.matmul(dfrac_wrapped, lattice)  # (A,B,3)

    # 거리 계산
    pair_dists = torch.norm(dr_mic, dim=-1).flatten()  # (A*B,)
    pair_dists = pair_dists[pair_dists > 1e-8]  # 자기 자신 제외

    if pair_dists.numel() == 0:
        return torch.zeros_like(bin_centers_t, device=device)

    # Gaussian kernel 적용 (미분 가능)
    norm_factor = math.sqrt(2.0 * math.pi) * sigma
    diff_r = pair_dists.unsqueeze(1) - bin_centers_t.unsqueeze(0)  # (P, M)
    kernel = torch.exp(-0.5 * (diff_r / sigma) ** 2) / norm_factor

    # bin별 합산
    prdf_unnorm = kernel.sum(dim=0)  # (M,)

    # 정규화 (volume, 원자수 등 고려)
    volume = float(atoms.get_volume())
    nA = len(idx1)
    nB = len(idx2)
    dr_val = float(step)
    r = bin_centers_t

    rho = nB / volume  # elem2의 밀도 (Å^-3)
    denom = (nA * rho * (4.0 * math.pi * (r ** 2) * dr_val)).to(device)
    denom = torch.where(denom == 0, torch.ones_like(denom, device=device), denom)
    prdf = prdf_unnorm / denom

    return prdf
