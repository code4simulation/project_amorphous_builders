# ==============================================================================
# Merged Python Script for project_amorphous_builders
#
# This script combines the code from the following modules for easier review:
# - src/main.py
# - src/utils/config.py
# - src/utils/visualization.py
# - src/utils/rdf_utils.py
# - src/data/loader.py
# - src/data/preprocessing.py
# - src/data/dataset.py
# - src/model/graph_network.py
# - src/model/diffusion.py
# - src/train.py
# - src/generate.py
# ==============================================================================

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
from ase.io import read, write
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# --- SECTION: UTILS (from src/utils/) ---
# ==============================================================================

# --- from src/utils/config.py ---
def load_config(config_path: str) -> Dict:
    """YAML 설정 파일을 로드하여 딕셔너리로 반환합니다."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"성공적으로 '{config_path}' 에서 설정을 로드했습니다.")
        return config
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{config_path}'를 찾을 수 없습니다.")
        exit(1)
    except Exception as e:
        print(f"오류: 설정 파일 로드 중 에러 발생: {e}")
        exit(1)

# --- from src/utils/visualization.py ---
def plot_loss_curve(losses: List[float], save_path: str):
    """학습 손실 곡선을 그리고 파일로 저장합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"학습 곡선을 '{save_path}'에 저장했습니다.")
    plt.close()

def plot_rdf_comparison(target_rdf: np.ndarray, generated_rdf: np.ndarray, save_path: str):
    """목표 RDF와 생성된 RDF를 비교하는 플롯을 저장합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(target_rdf[:, 0], target_rdf[:, 1], 'r--', label='Target RDF')
    plt.plot(generated_rdf[:, 0], generated_rdf[:, 1], 'b-', label='Generated RDF')
    plt.xlabel('r (Å)')
    plt.ylabel('g(r)')
    plt.title('RDF Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"RDF 비교 플롯을 '{save_path}'에 저장했습니다.")
    plt.close()

# --- from src/utils/rdf_utils.py ---
def differentiable_partial_rdf(atoms: Atoms, r_max: float, n_bins: int, elements: Tuple[str, str]) -> np.ndarray:
    """
    (Placeholder) 주어진 원자 구조에 대해 미분 가능한 부분 RDF를 계산합니다.
    실제 구현에서는 torch 연산을 사용하여 미분 가능성을 확보해야 합니다.
    """
    # ase.neighborlist와 numpy를 사용한 비-미분 버전의 예시
    from ase.neighborlist import neighbor_list
    
    indices_i = np.array([i for i, a in enumerate(atoms) if a.symbol == elements[0]])
    indices_j = np.array([j for j, a in enumerate(atoms) if a.symbol == elements[1]])
    
    if len(indices_i) == 0 or len(indices_j) == 0:
        return np.column_stack([np.linspace(0, r_max, n_bins), np.zeros(n_bins)])

    # 중심 원자(i)와 이웃 원자(j) 간의 거리 계산
    dists = atoms.get_distances(indices_i[:, np.newaxis], indices_j, mic=True).flatten()
    
    # 히스토그램 생성
    hist, bin_edges = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
    
    r = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    volume = 4.0 / 3.0 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    # 전체 밀도 계산
    total_volume = atoms.get_volume()
    n_j = len(indices_j)
    density = n_j / total_volume

    g_r = hist / (len(indices_i) * volume * density)
    
    # Si-Si 같은 쌍일 때 자기 자신과의 거리(0) 제외
    if elements[0] == elements[1]:
        g_r = hist / (len(indices_i) * volume * density)
    
    return np.column_stack([r, g_r])

def _calculate_cost(current_prdfs: Dict[str, np.ndarray], target_prdfs: Dict[str, np.ndarray]) -> float:
    """현재 pRDF와 목표 pRDF 간의 평균 제곱 오차(MSE)를 계산합니다."""
    total_error = 0.0
    num_pairs = 0
    for pair_key, target_rdf in target_prdfs.items():
        if pair_key in current_prdfs:
            current_rdf = current_prdfs[pair_key]
            if current_rdf.shape == target_rdf.shape:
                error = np.sum((current_rdf[:, 1] - target_rdf[:, 1])**2)
                total_error += error
                num_pairs += 1
    return total_error / num_pairs if num_pairs > 0 else 0.0

def reverse_monte_carlo_prdf(atoms: Atoms, prdfs: Dict[str, np.ndarray], n_steps: int, dr_max: float, temperature: float, r_max: float, n_bins: int, verbose: bool = False) -> Atoms:
    """RMC 방법을 사용하여 목표 pRDF에 맞도록 원자 구조를 최적화합니다."""
    current_atoms = atoms.copy()
    current_atoms.set_pbc(True)
    num_atoms = len(current_atoms)
    element_pairs = [tuple(key.split('-')) for key in prdfs.keys()]
    
    current_prdfs = {f"{p[0]}-{p[1]}": differentiable_partial_rdf(current_atoms, r_max, n_bins, p) for p in element_pairs}
    current_cost = _calculate_cost(current_prdfs, prdfs)
    
    if verbose: print(f"초기 비용: {current_cost:.6f}")

    for step in range(n_steps):
        atom_idx = random.randint(0, num_atoms - 1)
        old_position = current_atoms.positions[atom_idx].copy()
        
        displacement = (np.random.rand(3) - 0.5) * 2 * dr_max
        current_atoms.positions[atom_idx] += displacement
        current_atoms.wrap()

        new_prdfs = {f"{p[0]}-{p[1]}": differentiable_partial_rdf(current_atoms, r_max, n_bins, p) for p in element_pairs}
        new_cost = _calculate_cost(new_prdfs, prdfs)
        
        delta_cost = new_cost - current_cost
        accept = delta_cost < 0 or (temperature > 0 and random.random() < np.exp(-delta_cost / temperature))
        
        if accept:
            current_cost = new_cost
        else:
            current_atoms.positions[atom_idx] = old_position
            
        if verbose and (step + 1) % 100 == 0:
            print(f"스텝 [{step+1}/{n_steps}], 비용: {current_cost:.6f}, 수용 여부: {accept}")

    if verbose: print(f"최종 비용: {current_cost:.6f}")
    return current_atoms

# ==============================================================================
# --- SECTION: DATA HANDLING (from src/data/) ---
# ==============================================================================

# --- from src/data/loader.py ---
def load_atomic_data(file_path: str) -> List[Atoms]:
    """extxyz와 같은 원자 구조 파일을 읽어 ASE Atoms 객체 리스트로 반환합니다."""
    try:
        atoms_list = read(file_path, index=':')
        print(f"'{file_path}'에서 {len(atoms_list)}개의 구조를 로드했습니다.")
        return atoms_list
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return []

# --- from src/data/preprocessing.py ---
def preprocess_data(atoms_list: List[Atoms], normalize: bool = True) -> List[torch.Tensor]:
    """ASE Atoms 객체를 받아 모델 입력에 적합한 텐서로 변환합니다."""
    processed_data = []
    for atoms in atoms_list:
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        if normalize:
            # 셀 벡터를 기준으로 위치 정규화
            cell = torch.tensor(atoms.get_cell(), dtype=torch.float32)
            positions = torch.linalg.solve(cell.T, positions.T).T
            positions = positions % 1.0 # 주기 경계 조건
        processed_data.append(positions)
    return processed_data

# --- from src/data/dataset.py ---
class AtomicStructureDataset(Dataset):
    """원자 구조 데이터셋을 위한 PyTorch Dataset 클래스."""
    def __init__(self, data_tensors: List[torch.Tensor]):
        self.data = data_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==============================================================================
# --- SECTION: MODEL (from src/model/) ---
# ==============================================================================

# --- from src/model/graph_network.py ---
class GraphUNet(nn.Module):
    """(Placeholder) 그래프 U-Net 모델 정의."""
    def __init__(self, in_features, out_features):
        super(GraphUNet, self).__init__()
        # 간단한 MLP 예시. 실제로는 GNN 레이어(GCN, GAT 등)가 필요합니다.
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )
        print("GraphUNet 모델이 초기화되었습니다 (Placeholder).")

    def forward(self, x, t, edge_index=None):
        # t (timestep) 정보를 입력에 추가
        time_emb = t.float().view(-1, 1) # 간단한 시간 임베딩
        x_with_time = torch.cat([x, time_emb.expand(x.size(0), -1)], dim=1)
        return self.net(x_with_time)

# --- from src/model/diffusion.py ---
class DiffusionModel:
    """디퓨전 프로세스를 관리하는 클래스."""
    def __init__(self, denoise_net: nn.Module, n_timesteps: int, device: str):
        self.denoise_net = denoise_net
        self.n_timesteps = n_timesteps
        self.device = device
        
        # 선형 스케줄러 (Linear noise schedule)
        betas = torch.linspace(1e-4, 0.02, n_timesteps, device=device)
        alphas = 1. - betas
        self.alpha_hats = torch.cumprod(alphas, dim=0)
        print("DiffusionModel이 초기화되었습니다.")

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: 원본 데이터에 노이즈를 추가합니다."""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hats[t]).view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def sample(self, n_samples: int, atom_counts: int, atom_dims: int) -> torch.Tensor:
        """Reverse process: 노이즈로부터 새로운 샘플을 생성합니다."""
        print("샘플링을 시작합니다...")
        self.denoise_net.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, atom_counts, atom_dims, device=self.device)
            for t in range(self.n_timesteps - 1, -1, -1):
                time_tensor = torch.tensor([t] * n_samples, device=self.device)
                predicted_noise = self.denoise_net(x, time_tensor)
                
                alpha_t = 1. - torch.linspace(1e-4, 0.02, self.n_timesteps, device=self.device)[t]
                alpha_hat_t = self.alpha_hats[t]
                
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise)
                
                if t > 0:
                    z = torch.randn_like(x)
                    beta_t = torch.linspace(1e-4, 0.02, self.n_timesteps, device=self.device)[t]
                    x += torch.sqrt(beta_t) * z
        print("샘플링이 완료되었습니다.")
        self.denoise_net.train()
        return x

# ==============================================================================
# --- SECTION: TRAINING & GENERATION (from src/train.py, src/generate.py) ---
# ==============================================================================

# --- from src/train.py ---
def train_model(config: Dict, diffusion_model: DiffusionModel, dataset: AtomicStructureDataset):
    """모델 학습 루프를 제어합니다."""
    train_cfg = config['train']
    optimizer = torch.optim.Adam(diffusion_model.denoise_net.parameters(), lr=train_cfg['learning_rate'])
    loss_fn = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    device = diffusion_model.device
    
    losses = []
    print("모델 학습을 시작합니다...")
    for epoch in range(train_cfg['epochs']):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            x_0 = batch.to(device) # 원자 좌표
            t = torch.randint(0, diffusion_model.n_timesteps, (x_0.size(0),), device=device)
            
            x_t, noise = diffusion_model.add_noise(x_0, t)
            
            # 여기서 x_t를 그래프로 변환하고 GNN에 입력해야 하지만, 간단한 MLP 예시를 사용합니다.
            # 실제로는 x_t (좌표)와 원자 종류로부터 그래프(노드, 엣지)를 구성해야 합니다.
            # 이 예제에서는 좌표 텐서 자체를 GNN 입력으로 가정합니다.
            predicted_noise = diffusion_model.denoise_net(x_t, t)
            
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{train_cfg['epochs']}], Loss: {avg_loss:.6f}")
    
    print("모델 학습이 완료되었습니다.")
    
    # 모델과 손실 곡선 저장
    torch.save(diffusion_model.denoise_net.state_dict(), config['model']['save_path'])
    print(f"학습된 모델을 '{config['model']['save_path']}'에 저장했습니다.")
    plot_loss_curve(losses, config['results']['loss_curve_path'])

# --- from src/generate.py ---
def generate_structures(config: Dict, diffusion_model: DiffusionModel) -> List[Atoms]:
    """학습된 모델을 사용하여 새로운 원자 구조를 생성합니다."""
    gen_cfg = config['generate']
    
    # 모델 가중치 로드
    try:
        diffusion_model.denoise_net.load_state_dict(torch.load(config['model']['load_path']))
        diffusion_model.denoise_net.to(diffusion_model.device)
        print(f"'{config['model']['load_path']}'에서 모델 가중치를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{config['model']['load_path']}'를 찾을 수 없습니다. 먼저 모델을 학습시켜주세요.")
        exit(1)
        
    generated_coords_norm = diffusion_model.sample(
        n_samples=gen_cfg['num_samples'],
        atom_counts=config['structure']['atom_counts'],
        atom_dims=3
    )
    
    # 정규화된 좌표를 실제 좌표로 변환하고 ASE Atoms 객체 생성
    cell_matrix = np.array(config['structure']['cell'])
    atomic_symbols = config['structure']['symbols']
    
    generated_atoms_list = []
    for i in range(generated_coords_norm.size(0)):
        coords_norm = generated_coords_norm[i].cpu().numpy()
        coords_real = np.dot(coords_norm, cell_matrix)
        atoms = Atoms(symbols=atomic_symbols, positions=coords_real, cell=cell_matrix, pbc=True)
        generated_atoms_list.append(atoms)
        
    # 생성된 구조 저장
    save_path = config['results']['generated_structures_path']
    write(save_path, generated_atoms_list)
    print(f"{len(generated_atoms_list)}개의 생성된 구조를 '{save_path}'에 저장했습니다.")
    return generated_atoms_list

# ==============================================================================
# --- SECTION: MAIN EXECUTION (from src/main.py) ---
# ==============================================================================

def main():
    """메인 실행 파이프라인."""
    parser = argparse.ArgumentParser(description="Amorphous Structure Generator using Diffusion Models")
    parser.add_argument("mode", choices=['train', 'generate'], help="실행 모드: 'train' 또는 'generate'")
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    # 1. 설정 로드
    config = load_config(args.config)
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    # 2. 모델 초기화
    # Placeholder: GNN 입력 피처 수는 (x,y,z) + time_embedding = 4
    # Placeholder: GNN 출력 피처 수는 노이즈 예측을 위한 3 (dx,dy,dz)
    denoise_net = GraphUNet(in_features=4, out_features=3).to(device)
    
    diffusion_model = DiffusionModel(
        denoise_net=denoise_net,
        n_timesteps=config['diffusion']['timesteps'],
        device=device
    )

    if args.mode == 'train':
        # 3a. 학습 데이터 준비
        print("학습 데이터를 준비합니다...")
        atoms_list = load_atomic_data(config['data']['train_path'])
        if not atoms_list: return
        
        preprocessed_data = preprocess_data(atoms_list, normalize=True)
        dataset = AtomicStructureDataset(preprocessed_data)
        
        # 4a. 모델 학습
        train_model(config, diffusion_model, dataset)
        
    elif args.mode == 'generate':
        # 3b. 구조 생성
        print("구조 생성을 시작합니다...")
        generate_structures(config, diffusion_model)
        
    print("프로세스가 완료되었습니다.")

if __name__ == '__main__':
    main()

