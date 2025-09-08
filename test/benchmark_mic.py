import time
import numpy as np
import torch
from ase import Atoms
from ase.geometry import get_distances

def ase_mic_benchmark(positions, cell, pbc, idx1, idx2):
    # ASE MIC 거리 계산 (Numpy 기반)
    t0 = time.time()
    dists = get_distances(
        positions[idx1], positions[idx2], cell=cell, pbc=pbc
    )[1]
    t1 = time.time()
    return dists, t1 - t0

def torch_mic_benchmark(pos1, pos2, cell, pbc, device):
    # Torch MIC (직접 구현)
    lattice = torch.tensor(cell, dtype=torch.float32, device=device)  # (3,3)
    inv_lat = torch.linalg.inv(lattice)  # (3,3)
    dr = pos2.unsqueeze(-2) - pos1.unsqueeze(-3)  # (A,B,3)
    dfrac = torch.matmul(dr, inv_lat)  # (A,B,3)
    pbc_mask = torch.tensor(pbc, device=device, dtype=torch.bool)  # (3,)

    dfrac_wrapped = dfrac.clone()
    if pbc_mask.any():
        for i in range(3):
            if pbc_mask[i]:
                dfrac_wrapped[..., i] = dfrac_wrapped[..., i] - torch.floor(dfrac_wrapped[..., i] + 0.5)
    dr_mic = torch.matmul(dfrac_wrapped, lattice)  # (A,B,3)
    dists = torch.norm(dr_mic, dim=-1)  # (A,B)
    return dists

def run_benchmark(N=1000, device="cpu"):
    # 랜덤 구조 생성
    np.random.seed(42)
    cell = np.eye(3) * 10.0  # cubic box
    pbc = [True, True, True]
    positions = np.random.uniform(0, 10, size=(N, 3))
    symbols = ["Si"] * N
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    # 인덱스: 두 그룹
    idx1 = np.arange(N//2)
    idx2 = np.arange(N//2, N)

    print(f"Benchmarking MIC for N={N} atoms (ASE vs Torch, device={device})")

    # ASE
    t0 = time.time()
    ase_dists, ase_time = ase_mic_benchmark(positions, cell, pbc, idx1, idx2)
    t1 = time.time()
    print(f"ASE MIC: shape={ase_dists.shape}, time={ase_time:.4f}s")

    # Torch
    pos1 = torch.tensor(positions[idx1], dtype=torch.float32, device=device)
    pos2 = torch.tensor(positions[idx2], dtype=torch.float32, device=device)
    t2 = time.time()
    torch_dists = torch_mic_benchmark(pos1, pos2, cell, pbc, device)
    t3 = time.time()
    print(f"Torch MIC: shape={torch_dists.shape}, time={t3-t2:.4f}s")

    # 결과 비교
    # torch_dists와 ase_dists 값 비교 (CPU에서만, tolerance=1e-4)
    if device == "cpu":
        diff = np.abs(torch_dists.cpu().numpy() - ase_dists)
        print(f"Mean abs diff: {diff.mean():.6f}, Max abs diff: {diff.max():.6f}")

if __name__ == "__main__":
    # CPU 테스트
    run_benchmark(N=2000, device="cpu")
    # CUDA 테스트 (선택, GPU 있을 때)
    if torch.cuda.is_available():
        run_benchmark(N=2000, device="cuda")
