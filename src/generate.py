# src/generate.py
"""
Generate module with target-RDF handling and optional gradient-guidance during sampling.

Expected target_rdf_npy format: numpy array shape (M,2) columns: [r_bin_center, g(r)]
"""
import numpy as np
import torch
import yaml
from ase import Atoms
from src.utils import rdf_utils

def load_target_rdf(rdf_path):
    """
    npy 파일에서 target rdf (bin, rdf) 배열을 로드합니다.
    """
    arr = np.load(rdf_path)  # shape: (N, 2)
    bins = arr[:,0]
    rdf = arr[:,1]
    return bins, rdf

def parse_input_yaml(yaml_path):
    """
    input.yaml 파일에서 파라미터 로드
    """
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)
    density = params["density"]
    formula = params["formula"]
    lattice_vector = params["lattice_vector"]
    target_rdf_path = params["target_rdf_path"]
    output_path = params.get("output_path", "generated_structure.xyz")
    return density, formula, lattice_vector, target_rdf_path, output_path

def initialize_atoms(n_atoms, elem_counts, lattice_vector):
    """
    원자수, 원소 개수, 격자벡터로 Atoms 객체 초기 생성
    """
    # 화학종 리스트 생성
    symbols = []
    for elem, count in elem_counts.items():
        symbols.extend([elem]*count)
    # 전체 원자수에 맞게 채움
    while len(symbols) < n_atoms:
        for elem in elem_counts:
            if len(symbols) < n_atoms:
                symbols.append(elem)
    symbols = symbols[:n_atoms]
    symbols = np.array(symbols)
    # 임의 원자 좌표 생성
    lattice_vec = np.array(lattice_vector)
    positions = np.random.rand(n_atoms, 3) @ lattice_vec  # (n_atoms, 3)
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice_vec, pbc=True)
    return atoms

def generate_structure(
    density_g_cm3,
    formula,
    lattice_vector,
    target_rdf_path,
    model,
    device='cpu',
    output_path='generated_structure.xyz'
):
    """
    density, formula, lattice_vector, target_rdf를 받아 모델을 통해 구조 생성, 결과 저장
    """
    bins, target_rdf = load_target_rdf(target_rdf_path)
    n_atoms, elem_counts = rdf_utils.calculate_n_atoms(density_g_cm3, formula, lattice_vector)

    atoms = initialize_atoms(n_atoms, elem_counts, lattice_vector)

    # target RDF torch 텐서 변환
    bins_t = torch.tensor(bins, dtype=torch.float32, device=device)
    target_rdf_t = torch.tensor(target_rdf, dtype=torch.float32, device=device)

    # 모델 샘플링
    generated_atoms = model.generate(
        initial_atoms=atoms,
        target_bins=bins_t,
        target_rdf=target_rdf_t,
        lattice_vector=np.array(lattice_vector),
        device=device
    )

    # ASE xyz 저장
    generated_atoms.write(output_path, format='extxyz')
    print(f"Generated structure saved: {output_path}")

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


def main(input_yaml_path, model, device='cpu'):
    density, formula, lattice_vector, target_rdf_path, output_path = parse_input_yaml(input_yaml_path)
    generate_structure(
        density_g_cm3=density,
        formula=formula,
        lattice_vector=lattice_vector,
        target_rdf_path=target_rdf_path,
        model=model,
        device=device,
        output_path=output_path
    )

if __name__ == "__main__":
    import sys
    # input.yaml 경로와 모델 (사용자 구현 필요)
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "input.yaml"
    # from src.model import load_model
    # model = load_model(...)
    model = ... # 실제 모델 객체로 교체 필요
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(yaml_path, model, device=device)
