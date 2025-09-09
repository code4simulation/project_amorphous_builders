import torch 
import numpy as np

class ConditionalDiffusionModelWithCond(torch.nn.Module):
    def __init__(self, feature_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(feature_dim + cond_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, feature_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x, cond):
        # x : (batch, atoms, features)
        # cond : (batch, cond_dim) -> 확산 모델 조건, 배치 별로 같은 값 반복 사용 예정
        cond_expanded = cond.unsqueeze(1).repeat(1, x.size(1), 1) # (batch, atoms, cond_dim)
        inp = torch.cat([x, cond_expanded], dim=-1)
        h = self.relu(self.fc1(inp))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out


def generate_structure(
    model,
    acsf_feature_shape,
    atom_density,
    chem_formula,
    lattice_vectors,
    cond_scalers,
    num_steps=50,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    조건부 확산모델로 새로운 ACSF 기반 구조 생성

    Args:
        model (torch.nn.Module): 학습된 conditional diffusion model
        acsf_feature_shape (tuple): (원자수, 피처수)
        atom_density (float): 원자밀도 (예: atoms per volume)
        chem_formula (dict): {원소명: 개수} 예: {'Si': 10, 'O': 20}
        lattice_vectors (np.ndarray): (3,3) lattice 벡터 배열
        cond_scalers (dict): 조건값 스케일러, 조건 텐서 스케일링 및 전처리용
        num_steps (int): 역확산 단계 수
        device: torch device

    Returns:
        generated_acsf_features (torch.Tensor): (원자수, 피처수) 생성된 ACSF 벡터
    """

    model.eval()
    with torch.no_grad():
        atoms_num, feat_dim = acsf_feature_shape

        # 조건 벡터 구성 예시 (원자밀도, 화학식 숫자 벡터, lattice 벡터 전개)
        # 화학식 원소 리스트는 cond_scalers['element_list']로 받아온다고 가정
        cond_elems = cond_scalers['element_list'] # e.g. ['Si', 'O']
        
        # 화학식 숫자 벡터화 (각 원소 개수/총 원자 수 비율)
        chem_vec = torch.zeros(len(cond_elems), device=device)
        total_atoms = sum(chem_formula.values())
        for i, elem in enumerate(cond_elems):
            chem_vec[i] = chem_formula.get(elem, 0) / total_atoms

        # lattice vector flatten 및 normalize (스케일러 필요시 cond_scalers 활용)
        lat_vec_flat = torch.tensor(lattice_vectors.flatten(), dtype=torch.float32, device=device)
        # 스케일링(있으면 cond_scalers['lattice']) 적용. 없으면 그대로 사용
        if 'lattice' in cond_scalers:
            lat_vec_flat = (lat_vec_flat - cond_scalers['lattice']['mean']) / cond_scalers['lattice']['std']

        # atom_density 스케일링
        if 'atom_density' in cond_scalers:
            atom_dens = torch.tensor([(atom_density - cond_scalers['atom_density']['mean']) /
                                      cond_scalers['atom_density']['std']], device=device)
        else:
            atom_dens = torch.tensor([atom_density], device=device)

        # 조건 벡터 (합치기)
        cond_vec = torch.cat([atom_dens, chem_vec, lat_vec_flat], dim=0)
        cond_vec = cond_vec.unsqueeze(0) # batch 차원 추가 (1, cond_dim)

        # 초기 노이즈 텐서 샘플링(가우시안)
        x_t = torch.randn(1, atoms_num, feat_dim, device=device)

        # 역확산 생성 과정 단순화 예시
        for t in reversed(range(num_steps)):
            noise_pred = model(x_t, cond_vec) # 예측 노이즈
            # 노이즈 제거 (간단한 스텝 - 실제 DDPM은 스케줄러 필요)
            x_t = x_t - noise_pred / num_steps
            # optional: 노이즈 주입, 클리핑 처리 등 추가 가능

        generated_acsf_features = x_t.squeeze(0).cpu() # (atoms_num, feature_dim)

    return generated_acsf_features


# 사용 예시

if __name__ == "__main__":
    # 학습 완료된 모델 불러오기
    model = ConditionalDiffusionModelWithCond(feature_dim=64, cond_dim=10) # 예제 차원
    model.load_state_dict(torch.load("best_cond_diffusion.pth"))
    model.eval()

    # 조건 정보 예시
    atom_density = 2.32 # 단위 자유롭게, 스케일에 맞게 조정 필요
    chem_formula = {'Si': 100}
    lattice_vectors = np.eye(3) * 10.0 # 단순 10 Å 큐브 구조 예시

    # cond_scalers 예시 (실제 환경에 맞게 미리 계산하거나 로드)
    cond_scalers = {
        'element_list': ['Si'],
        'atom_density': {'mean': 2.32, 'std': 0.01},
        'lattice': {'mean': 12.0, 'std': 2.0}
    }

    # ACSF feature shape 예시
    atoms_num = sum(chem_formula.values())
    feature_dim = 64 # ACSF feature 차원 수

    generated_features = generate_structure(
        model=model,
        acsf_feature_shape=(atoms_num, feature_dim),
        atom_density=atom_density,
        chem_formula=chem_formula,
        lattice_vectors=lattice_vectors,
        cond_scalers=cond_scalers,
        num_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Generated ACSF features shape:", generated_features.shape)
