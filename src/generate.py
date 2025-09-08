"""
Generate mode implementation.

This module provides structure generation from conditional inputs such as RDF,
lattice vector, chemical formula, and number of atoms.

Outputs:
    - final.extxyz : always generated
    - denoising.extxyz : optional, when denoising=True
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple


def write_extxyz(path: str, elements: List[str], positions: np.ndarray, comment: str = ""):
    """
    Write atomic structure to an extxyz file.

    Args:
        path: file path to save
        elements: list of atomic symbols
        positions: (N, 3) array of coordinates
        comment: comment line (metadata)
    """
    n_atoms = len(elements)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n_atoms}\n")
        f.write(comment + "\n")
        for el, pos in zip(elements, positions):
            f.write(f"{el} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def parse_formula(formula: str, n_atoms: int) -> List[str]:
    """
    Parse chemical formula and distribute element counts across n_atoms.

    Args:
        formula: chemical formula, e.g. "SiO2"
        n_atoms: number of atoms to generate

    Returns:
        list of element symbols of length n_atoms
    """
    import re
    token_re = re.compile(r"([A-Z][a-z]*)(\d*)")
    tokens = token_re.findall(formula)
    if not tokens:
        return ["X"] * n_atoms

    parsed = []
    total_ratio = 0
    for el, num in tokens:
        count = int(num) if num else 1
        parsed.append((el, count))
        total_ratio += count

    elements = []
    accumulated = 0
    for i, (el, cnt) in enumerate(parsed):
        if i == len(parsed) - 1:
            n_el = n_atoms - accumulated
        else:
            n_el = int(round(n_atoms * (cnt / total_ratio)))
            accumulated += n_el
        elements.extend([el] * n_el)

    if len(elements) < n_atoms:
        elements.extend([parsed[0][0]] * (n_atoms - len(elements)))
    return elements[:n_atoms]


def denoise_positions(positions: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """
    Simple denoising by local averaging.
    """
    n = positions.shape[0]
    out = positions.copy()
    r = sigma * 3.0
    for i in range(n):
        pi = positions[i]
        dists = np.linalg.norm(positions - pi, axis=1)
        mask = dists <= r
        if mask.sum() > 1:
            out[i] = positions[mask].mean(axis=0)
    return out


def generate_from_cond(
    rdf,
    lattice_vector: np.ndarray,
    formula: str,
    n_atoms: int,
    output_dir: str = ".",
    denoise: bool = True,
    model=None,
    model_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None
) -> Tuple[str, Optional[str]]:
    """
    Generate atomic structure from conditional inputs.

    Args:
        rdf: RDF array or file path
        lattice_vector: 3x3 lattice vector
        formula: chemical formula string
        n_atoms: number of atoms
        output_dir: directory to save outputs
        denoise: whether to apply denoising
        model: optional generative model with .sample(cond, **kwargs)
        model_kwargs: extra arguments for model.sample
        seed: random seed

    Returns:
        final_path, denoise_path
    """
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    lattice_vector = np.asarray(lattice_vector, dtype=float)
    if lattice_vector.shape != (3, 3):
        raise ValueError("lattice_vector must be shape (3,3)")

    cond = {
        "rdf": str(type(rdf)),
        "lattice_vector": lattice_vector.tolist(),
        "formula": formula,
        "n_atoms": n_atoms,
    }

    elements, positions = None, None
    if model is not None:
        if model_kwargs is None:
            model_kwargs = {}
        try:
            elements, positions = model.sample(cond, **model_kwargs)
        except Exception as e:
            print(f"[generate] Model failed, fallback. Error: {e}")

    if elements is None or positions is None:
        elements = parse_formula(formula, n_atoms)
        frac = np.random.rand(n_atoms, 3)
        positions = frac @ lattice_vector
        positions += (np.random.rand(n_atoms, 3) - 0.5) * 1e-3

    denoise_path = None
    if denoise:
        positions = denoise_positions(positions, sigma=0.2)
        denoise_path = os.path.join(output_dir, "denoising.extxyz")
        write_extxyz(denoise_path, elements, positions,
                     comment=json.dumps({"stage": "denoising", **cond}))

    final_path = os.path.join(output_dir, "final.extxyz")
    write_extxyz(final_path, elements, positions,
                 comment=json.dumps({"stage": "final", **cond}))

    return final_path, denoise_path
