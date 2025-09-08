# src/generate.py
"""
Generate module with target-RDF handling and optional gradient-guidance during sampling.

Expected target_rdf_npy format: numpy array shape (M,2) columns: [r_bin_center, g(r)]
"""
import os
import json
import numpy as np
from typing import Optional, Dict, Tuple
from ase import Atoms
from ase.io import write
import torch

from src.utils.rdf_utils import differentiable_rdf, estimate_n_atoms_from_density

def load_target_rdf(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Target RDF npy must be shape (M,2) with columns [r_bin, g_r]")
    bins = arr[:,0].astype(float)
    rdf = arr[:,1].astype(float)
    return bins, rdf

def parse_formula(formula: str, n_atoms: int):
    # simple parser for element assignment (fallback); keep in sync with project's style guide
    import re
    token_re = re.compile(r"([A-Z][a-z]*)(\d*)")
    tokens = token_re.findall(formula)
    if not tokens:
        return ["X"] * n_atoms
    parsed = []
    total = 0
    for el,num in tokens:
        cnt = int(num) if num else 1
        parsed.append((el,cnt))
        total += cnt
    elems = []
    running = 0
    for i,(el,cnt) in enumerate(parsed):
        if i == len(parsed)-1:
            n_el = n_atoms - running
        else:
            n_el = int(round(n_atoms * (cnt/total)))
            running += n_el
        elems.extend([el]*n_el)
    if len(elems) < n_atoms:
        elems.extend([parsed[0][0]]*(n_atoms-len(elems)))
    return elems[:n_atoms]

def generate_from_cond(
    target_rdf_npy: Optional[str],
    lattice_vector: np.ndarray,
    formula: str,
    n_atoms: Optional[int],
    density: Optional[float] = None,
    output_dir: str = ".",
    denoise: bool = True,
    model=None,
    model_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    rdf_guidance_weight: float = 0.0,
    rdf_guidance_steps: int = 1,
    rdf_guidance_lr: float = 1e-3
) -> Tuple[str, Optional[str]]:
    """
    Main generate function.

    - target_rdf_npy: path to npy (M,2): [r_bin, g(r)]
    - model: expected to be a sampler object exposing sample(cond, n_atoms, lattice_vector, device, **kwargs)
             which returns (elements_list, positions (N,3) numpy)
    - rdf_guidance_weight > 0 enables gradient-based guidance during sampling (see notes)
    """
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    lattice_vector = np.asarray(lattice_vector, dtype=float)
    if lattice_vector.shape != (3,3):
        raise ValueError("lattice_vector must be (3,3)")

    # compute n_atoms if needed
    if n_atoms is None:
        if density is None:
            raise ValueError("Either n_atoms or density must be provided")
        n_atoms = estimate_n_atoms_from_density(formula, density, lattice_vector)
        print(f"[generate] estimated n_atoms={n_atoms}")

    # load target rdf
    target_bins = None
    target_rdf = None
    if target_rdf_npy is not None:
        target_bins, target_rdf = load_target_rdf(target_rdf_npy)
        target_bins_t = torch.tensor(target_bins, dtype=torch.float32, device=device)
        target_rdf_t = torch.tensor(target_rdf, dtype=torch.float32, device=device)
    else:
        target_bins_t = None
        target_rdf_t = None

    cond = {"lattice_vector": lattice_vector.tolist(), "formula": formula, "n_atoms": int(n_atoms)}
    if target_rdf is not None:
        cond["rdf"] = target_rdf.tolist()
        cond["rdf_bins"] = target_bins.tolist()

    # Sample with model
    elements, positions = None, None
    if model is not None and hasattr(model, "sample"):
        # we expect model.sample to accept cond, n_atoms, lattice_vector, device, and optionally target guidance params
        try:
            # if model can implement internal guidance, pass target_rdf; otherwise we'll post-process
            sample_out = model.sample(cond=cond, n_atoms=n_atoms, lattice_vector=lattice_vector, device=device, **(model_kwargs or {}))
            if isinstance(sample_out, tuple) and len(sample_out) == 2:
                elements, positions = sample_out
                positions = np.asarray(positions, dtype=float)
        except Exception as e:
            print(f"[generate] model.sample failed: {e}")

    # fallback random
    if elements is None or positions is None:
        elements = parse_formula(formula, n_atoms)
        frac = np.random.rand(n_atoms,3)
        positions = frac @ lattice_vector
        positions += (np.random.rand(n_atoms,3)-0.5)*1e-3

    # if user requested local guidance after sampling (refinement), run a few gradient steps
    if rdf_guidance_weight > 0.0 and target_rdf_t is not None:
        # convert to torch
        pos_t = torch.tensor(positions, dtype=torch.float32, device=device, requires_grad=True)
        bins_t = target_bins_t
        for step in range(rdf_guidance_steps):
            g_pred = differentiable_rdf(pos_t, bins_t, sigma=0.1, lattice=lattice_vector, device=device)
            rdf_loss = torch.nn.functional.mse_loss(g_pred, target_rdf_t)
            loss = rdf_guidance_weight * rdf_loss
            # gradient step on positions (descent)
            grads = torch.autograd.grad(loss, pos_t)[0]
            pos_t = (pos_t - rdf_guidance_lr * grads).detach().requires_grad_()
        positions = pos_t.detach().cpu().numpy()

    # create ASE atoms and save
    atoms = Atoms(symbols=elements, positions=positions, cell=lattice_vector, pbc=True)
    atoms.info = {"stage": "generated", **cond}
    denoise_path = None
    if denoise and target_rdf is not None:
        # compute predicted rdf for inspection, save as denoising.extxyz with atoms.info containing pred rdf
        pos_t = torch.tensor(positions, dtype=torch.float32, device=device)
        bins_t = torch.tensor(target_bins, dtype=torch.float32, device=device)
        pred_g = differentiable_rdf(pos_t, bins_t, sigma=0.1, lattice=lattice_vector, device=device)
        atoms.info["pred_rdf_bins"] = target_bins.tolist()
        atoms.info["pred_rdf"] = pred_g.detach().cpu().numpy().tolist()
        denoise_path = os.path.join(output_dir, "denoising.extxyz")
        write(denoise_path, atoms, format="extxyz")

    final_path = os.path.join(output_dir, "final.extxyz")
    atoms.info["stage"] = "final"
    write(final_path, atoms, format="extxyz")
    return final_path, (denoise_path if denoise and target_rdf is not None else None)
