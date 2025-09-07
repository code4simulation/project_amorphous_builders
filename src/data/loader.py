import logging
import numpy as np
from ase.io import read
from ase import Atoms
import torch
from torch_geometric.data import Data
from typing import List, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_extxyz(file_path: str) -> List[Atoms]:
    """
    extxyz 파일에서 원자 구조를 로드합니다.
    
    Args:
        file_path (str): extxyz 파일 경로
        
    Returns:
        List[Atoms]: 원자 구조 리스트
    """
    try:
        structures = read(file_path, index=':')
        logger.info(f"성공적으로 {len(structures)}개의 구조를 로드했습니다.")
        return structures
    except Exception as e:
        logger.error(f"파일 로드 중 오류 발생: {str(e)}")
        raise

def atoms_to_graph(
    atoms: Atoms, 
    graph_cutoff: float = 5.0,
    node_features: Optional[List[str]] = None
) -> Data:
    """
    ASE Atoms 객체를 PyG 그래프로 변환합니다.
    
    Args:
        atoms (Atoms): 변환할 Atoms 객체
        graph_cutoff (float): 그래프 구성 시 cutoff distance (Å)
        node_features (List[str]): 노드 특징으로 사용할 속성 목록
        
    Returns:
        Data: PyG 그래프 데이터 객체
    """
    if node_features is None:
        node_features = ['atomic_number', 'positions']
    
    try:
        # 노드 특징 추출
        node_attrs = []
        atomic_numbers = None
        
        for feature in node_features:
            if feature == 'atomic_number':
                # 원자 번호를 별도로 저장 (임베딩 레이어에서 사용)
                atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            elif feature == 'positions':
                node_attrs.append(torch.tensor(atoms.positions, dtype=torch.float))
            # 추가 특징들 확장 가능
        
        # 노드 특징 결합 (positions만 사용)
        x = torch.cat(node_attrs, dim=1) if len(node_attrs) > 1 else node_attrs[0]
        
        # 에지 인덱스 및 속성 계산
        positions = atoms.positions
        num_atoms = len(atoms)
        
        # 거리 행렬 계산
        dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        
        # cutoff 내의 에지 필터링
        edge_mask = (dist_matrix > 0) & (dist_matrix <= graph_cutoff)
        edge_index = np.array(np.where(edge_mask))
        
        # 에지 속성 (거리)
        edge_attr = dist_matrix[edge_mask].reshape(-1, 1)
        
        # PyG 데이터 객체 생성
        graph_data = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),  # 반드시 포함!
            num_nodes=num_atoms
        )
        
        # 원자 번호를 그래프 데이터에 추가
        if atomic_numbers is not None:
            graph_data.atomic_numbers = atomic_numbers
        
        logger.debug(f"그래프 변환 완료: {num_atoms}개 원자, {edge_index.shape[1]}개 에지")
        return graph_data
        
    except Exception as e:
        logger.error(f"그래프 변환 중 오류 발생: {str(e)}")
        raise


def batch_structures_to_graphs(
    structures: List[Atoms], 
    graph_cutoff: float = 5.0
) -> List[Data]:
    """
    Atoms 구조 배치를 그래프 데이터 리스트로 변환합니다.
    
    Args:
        structures (List[Atoms]): Atoms 객체 리스트
        graph_cutoff (float): 그래프 구성 시 cutoff distance (Å)
        
    Returns:
        List[Data]: PyG 그래프 데이터 리스트
    """
    graphs = []
    for i, atoms in enumerate(structures):
        try:
            graph = atoms_to_graph(atoms, graph_cutoff)
            graphs.append(graph)
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}개 구조 변환 완료")
        except Exception as e:
            logger.warning(f"구조 {i} 변환 실패: {str(e)}")
            continue
    
    logger.info(f"총 {len(graphs)}개 그래프 변환 완료")
    return graphs
