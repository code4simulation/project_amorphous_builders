import torch
from torch_geometric.data import Data
from e3nn import o3
from ase.io import read, iread
from ase.geometry import get_distances
import numpy as np

def preprocess_extxyz_with_e3nn(atoms, r_max=6.0, lmax=2):
    """
    NequIP 스타일의 e3nn 기반 전처리 함수 (PBC 지원)
    
    Args:
        file_path (str): .extxyz 파일 경로
        r_max (float): 최대 거리 커트오프 (Å)
        lmax (int): spherical harmonics의 최대 각운동량 양자수
    
    Returns:
        Data: torch_geometric Data 객체
    """
    num_atoms = len(atoms)
    cell = atoms.cell  # 단위 셀 벡터
    pbc = atoms.pbc    # 주기적 경계 조건 설정 [True, True, True]
    
    # 2. 원자 좌표와 종류 추출
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    
    # 3. PBC를 고려한 원자 간 거리와 벡터 계산
    # ASE의 get_distances는 PBC를 고려한 최소 이미지 컨벤션 적용
    dist_matrix, dist_vecs = get_distances(atoms.positions, cell=cell, pbc=pbc)
    
    # 4. 거리 커트오프 내의 원자 쌍 찾기 (자기 자신 제외)
    edge_indices = np.vstack(np.where((dist_matrix > 0) & (dist_matrix <= r_max)))
    
    # 간선이 없는 경우 처리
    if edge_indices.size == 0:
        # 빈 간선을 위한 더미 데이터 생성
        edge_indices = np.array([[0], [0]])
        edge_vecs = np.zeros((1, 3))
        edge_lengths = torch.zeros(1, 1)
        edge_directions = torch.zeros(1, 3)
    else:
        # 5. 간선 특징 계산 (spherical harmonics)
        edge_vecs = dist_vecs[edge_indices[0], edge_indices[1]]
        edge_vecs = torch.tensor(edge_vecs, dtype=torch.float32)
        
        # 거리 계산 (마지막 차원에 대해 norm 계산)
        edge_lengths = torch.norm(edge_vecs, dim=-1, keepdim=True)
        
        # edge_lengths가 0인 경우 방지 (NaN 방지)
        edge_lengths = torch.clamp(edge_lengths, min=1e-8)
        
        edge_directions = edge_vecs / edge_lengths
    
    # 패리티 문제 해결: l=1에 대해 o(odd)만 사용
    # l=0: e(even), l=1: o(odd), l=2: e(even) ...
    irrep_str = "1x0e"  # l=0
    for l in range(1, lmax+1):
        if l % 2 == 1:  # 홀수 l은 odd 패리티
            irrep_str += f"+1x{l}o"
        else:  # 짝수 l은 even 패리티
            irrep_str += f"+1x{l}e"
    irreps = o3.Irreps(irrep_str)
    
    # e3nn의 spherical harmonics 계산 (정규화 포함)
    try:
        sph_harm = o3.spherical_harmonics(irreps, edge_directions, normalize=True)
    except Exception as e:
        print(f"Spherical harmonics 계산 오류: {e}")
        # 오류 발생 시 기본값 사용
        sph_harm = torch.ones(edge_directions.shape[0], irreps.dim)
    # 6. 노드 특징 준비 (원자 종류 임베딩용)
    # 실제 임베딩은 모델에서 수행되므로 여기서는 원자 번호만 사용
    node_features = atomic_numbers
    
    # 7. torch_geometric Data 객체 생성
    data = Data(
        x=node_features,            # 원자 번호 [num_nodes]
        pos=pos,                    # 원자 좌표 [num_nodes, 3]
        edge_index=torch.tensor(edge_indices, dtype=torch.long),  # 간선 [2, num_edges]
        edge_attr=sph_harm,         # spherical harmonics 특징 [num_edges, feature_dim]
        edge_length=edge_lengths,   # 간선 길이 [num_edges, 1]
        cell=torch.tensor(cell.array, dtype=torch.float32).reshape(3, 3),  # 단위 셀
        pbc=torch.tensor(pbc, dtype=torch.bool)  # PBC 설정
    )
    
    return data

# 사용 예제
if __name__ == "__main__":
    try:
        data = []
        for i, atoms in enumerate(iread('amorphous.extxyz', format='extxyz')):
            data.append(preprocess_extxyz_with_e3nn(atoms, r_max=5.0, lmax=2))
        
            print(f"노드 수: {data[-1].x.shape[0]}")
            print(f"간선 수: {data[-1].edge_index.shape[1]}")
            
            # edge_attr 차원 확인
            if hasattr(data[-1], 'edge_attr') and data[-1].edge_attr is not None:
                if data[-1].edge_attr.dim() == 1:
                    print(f"간선 특징 차원: 1 (스칼라 값)")
                    print(f"간선 특징 예시: {data[-1].edge_attr[:5]}")
                else:
                    print(f"간선 특징 차원: {data[-1].edge_attr.shape[1]}")
                    print(f"간선 특징 예시: {data[-1].edge_attr[:2, :5]}")
            else:
                print("간선 특징이 없습니다.")
                
            print(f"단위 셀: {data[-1].cell}")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
