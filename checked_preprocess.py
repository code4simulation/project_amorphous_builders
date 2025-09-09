import logging
import sys, os
import yaml
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
from ase.io import read, write
from ase.geometry import get_distances
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- SECTION: MAIN EXECUTION (from src/main.py) ---
# ==============================================================================
def main():
    config = load_config(sys.argv[1])
    mode = config.get('mode')

    if not config.validate():
        raise RuntimeError("Config validation failed")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == 'preprocess':
        extxyz_path = config.get('data.extxyz_path')
        dataset_path = config.get('data.dataset_path')
        graph_cutoff = config.get('data.graph_cutoff', 5.0)
        r_max = config.get('data.r_max', 5.0)
        n_bins = config.get('data.n_bins', 100)
        normalize_rdf = config.get('data.normalize_rdf', True)

        structures = load_extxyz(extxyz_path)
        dataset = create_dataset_from_structures(
            structures,
            graph_cutoff=graph_cutoff,
            r_max=r_max,
            n_bins=n_bins,
            normalize=normalize_rdf
        )
        dataset.save(dataset_path)
        print(f"Preprocessing complete: saved to {dataset_path}")
        return

    elif mode == 'train':
        pass        
    elif mode == 'generate':
        pass
    return 0

# ==============================================================================
# --- SECTION: UTILS (from src/utils/) ---
# ==============================================================================
# --- from src/utils/config.py ---
class Config:
    """YAML 설정 파일을 파싱하고 관리하는 클래스"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        try:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            except UnicodeDecodeError:
                with open(self.config_path, 'r', encoding='cp949') as f:
                    self.config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 완료: {self.config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            raise
        
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            logger.warning(f"설정 키를 찾을 수 없습니다: {key}, 기본값: {default}")
            return default
    
    def validate(self) -> bool:
        required_keys = []
        mode = self.get('mode')
        if mode == 'preprocess':
            required_keys = [
                'data.extxyz_path',
                'data.dataset_path'
            ]
        elif mode == 'train':
            required_keys = [
                'data.dataset_path',
                'model.node_dim',
                'model.hidden_dim',
                'training.batch_size',
                'training.num_epochs'
            ]
        elif mode == 'generate':
            required_keys = [
                'data.dataset_path',
                'model.node_dim',
                'model.hidden_dim',
            ]

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        if missing_keys:
            logger.error(f"필수 설정 값이 누락되었습니다: {missing_keys}")
            return False
        logger.info("설정 검증 완료")
        return True
    
    def update(self, key: str, value: Any):
        keys = key.split('.')
        config_dict = self.config
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        config_dict[keys[-1]] = value
        logger.debug(f"설정 업데이트: {key} = {value}")
    
    def save(self, path: Optional[str] = None):
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"설정 파일 저장 완료: {save_path}")
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False)

def load_config(config_path: str) -> Config:
    return Config(config_path)

# ==============================================================================
# --- SECTION: DATA HANDLING (from src/data/) ---
# ==============================================================================
# --- from src/data/loader.py ---
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

# --- from src/data/preprocessing.py ---
def calculate_rdf(
    atoms: Atoms, 
    r_max: float = 10.0, 
    n_bins: int = 200,
    elements: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radial Distribution Function (RDF)를 계산합니다.
    
    Args:
        atoms (Atoms): 원자 구조
        r_max (float): 최대 거리 (Å)
        n_bins (int): 히스토그램 빈 개수
        elements (List[str]): 특정 원소 쌍에 대한 부분 RDF 계산 (None이면 전체 RDF)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (거리 배열, RDF 값 배열)
    """
    try:
        positions = atoms.positions
        cell = atoms.cell
        pbc = atoms.pbc
        
        # 모든 원자 쌍 간의 거리 계산
        if np.any(pbc) and cell.rank == 3:
            # 주기성 경계 조건 적용
            dists = get_distances(positions, positions, cell=cell, pbc=pbc)[1]
            dists = dists.reshape(-1)
        else:
            # 비주기성 시스템
            dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            dists = dists.reshape(-1)
        
        # 0 거리 제외
        dists = dists[dists > 0]
        
        # 부분 RDF 계산 (특정 원소 쌍에 대해)
        if elements is not None and len(elements) == 2:
            element1, element2 = elements
            indices1 = [i for i, sym in enumerate(atoms.symbols) if sym == element1]
            indices2 = [i for i, sym in enumerate(atoms.symbols) if sym == element2]
            
            if not indices1 or not indices2:
                logger.warning(f"원소 {element1} 또는 {element2}가 구조에 존재하지 않습니다.")
                return np.zeros(n_bins), np.zeros(n_bins)
            
            # 특정 원소 쌍 간의 거리만 계산
            if np.any(pbc) and cell.rank == 3:
                dists = get_distances(positions[indices1], positions[indices2], 
                                     cell=cell, pbc=pbc)[1].reshape(-1)
            else:
                dists = np.linalg.norm(
                    positions[indices1, None, :] - positions[None, indices2, :], 
                    axis=-1
                ).reshape(-1)
            
            # 0 거리 제외
            dists = dists[dists > 0]
        
        # RDF 히스토그램 계산
        hist, bin_edges = np.histogram(dists, bins=n_bins, range=(0, r_max))
        r = (bin_edges[:-1] + bin_edges[1:]) / 2  # 빈 중심값
        
        # RDF 정규화
        volume = atoms.get_volume()
        n_atoms = len(atoms)
        dr = bin_edges[1] - bin_edges[0]
        
        # 이상 기체 RDF (4πr²ρdr)
        ideal_gas = 4 * np.pi * r**2 * (n_atoms / volume) * dr
        rdf = hist / ideal_gas / n_atoms
        
        logger.debug(f"RDF 계산 완료: 최대값 {np.max(rdf):.2f}, 평균 {np.mean(rdf):.2f}")
        return r, rdf
        
    except Exception as e:
        logger.error(f"RDF 계산 중 오류 발생: {str(e)}")
        raise

def normalize_rdf(rdf: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    RDF 값을 정규화합니다.
    
    Args:
        rdf (np.ndarray): 원본 RDF 배열
        method (str): 정규화 방법 ('minmax', 'standard', 'none')
        
    Returns:
        np.ndarray: 정규화된 RDF 배열
    """
    try:
        if method == 'minmax':
            rdf_min = np.min(rdf)
            rdf_max = np.max(rdf)
            if rdf_max - rdf_min > 1e-10:
                normalized = (rdf - rdf_min) / (rdf_max - rdf_min)
            else:
                normalized = np.zeros_like(rdf)
        elif method == 'standard':
            mean = np.mean(rdf)
            std = np.std(rdf)
            if std > 1e-10:
                normalized = (rdf - mean) / std
            else:
                normalized = np.zeros_like(rdf)
        elif method == 'none':
            normalized = rdf
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")
        
        logger.debug(f"RDF 정규화 완료: 방법 {method}")
        return normalized
        
    except Exception as e:
        logger.error(f"RDF 정규화 중 오류 발생: {str(e)}")
        raise

# --- from src/data/dataset.py ---
class AmorphousDataset(Dataset):
    def __init__(
        self, 
        graphs: List[Data],
        rdf_features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.graphs = graphs
        self.rdf_features = rdf_features
        self.metadata = metadata or {}
        self._validate_data()
        logger.info(f"데이터셋 초기화 완료: {len(self)}개 샘플")

    def _validate_data(self):
        if len(self.graphs) == 0:
            raise ValueError("빈 그래프 리스트")
        if self.rdf_features is not None and len(self.graphs) != len(self.rdf_features):
            raise ValueError("그래프와 RDF 특성의 개수가 일치하지 않습니다")
        logger.info("데이터 검증 완료")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        if not hasattr(graph, 'pos') or graph.pos is None:
            logger.error(f"샘플 {idx}에서 pos 필드가 None입니다. 데이터 생성/저장/로드 과정을 점검하세요.")
            raise ValueError(f"샘플 {idx}의 pos가 None입니다.")
        if self.rdf_features is not None:
            graph.rdf = torch.tensor(self.rdf_features[idx], dtype=torch.float)
        return graph

    def save(self, path: str):
        try:
            save_data = {
                'graphs': self.graphs,
                'rdf_features': self.rdf_features,
                'metadata': self.metadata
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(save_data, path)
            logger.info(f"데이터셋 저장 완료: {path}")
        except Exception as e:
            logger.error(f"데이터셋 저장 중 오류 발생: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'AmorphousDataset':
        try:
            save_data = torch.load(path)
            dataset = cls(
                graphs=save_data['graphs'],
                rdf_features=save_data.get('rdf_features'),
                metadata=save_data.get('metadata', {})
            )
            logger.info(f"데이터셋 로드 완료: {path}, {len(dataset)}개 샘플")
            return dataset
        except Exception as e:
            logger.error(f"데이터셋 로드 중 오류 발생: {str(e)}")
            raise

def create_dataset_from_structures(
    structures: List[Any],
    graph_cutoff: float = 5.0,
    r_max: float = 10.0,
    n_bins: int = 100,
    normalize: bool = True
) -> AmorphousDataset:
    """
    ASE Atoms 객체 리스트로부터 데이터셋 생성
    
    Args:
        structures: ASE Atoms 객체 리스트
        graph_cutoff: 그래프 구성 시 cutoff distance (Å)
        r_max: RDF 계산 최대 거리 (Å)
        n_bins: RDF 히스토그램 bin 개수
        normalize: RDF 정규화 여부
        
    Returns:
        AmorphousDataset: 변환된 데이터셋
    """
    
    logger.info("데이터셋 생성 시작")
    
    graphs = []
    rdf_features = []
    
    for i, atoms in enumerate(structures):
        try:
            graph = atoms_to_graph(atoms, graph_cutoff=graph_cutoff)
            graphs.append(graph)
            
            rdf = calculate_rdf(atoms, r_max=r_max, n_bins=n_bins)
            if normalize:
                rdf = normalize_rdf(rdf)
            rdf_features.append(rdf)
            
            if (i + 1) % 10 == 0:
                logger.info(f"{i + 1}/{len(structures)}개 구조 처리 완료")
                
        except Exception as e:
            logger.warning(f"구조 {i} 변환 실패: {str(e)}")
            continue
    
    rdf_features = np.array(rdf_features) if rdf_features else None
    
    logger.info(f"총 {len(structures)}개 구조 중 {len(graphs)}개 성공적으로 변환")
    
    metadata = {
        'graph_cutoff': graph_cutoff,
        'r_max': r_max,
        'n_bins': n_bins,
        'normalize': normalize,
        'source_structures_count': len(structures),
        'converted_structures_count': len(graphs)
    }
    return AmorphousDataset(graphs, rdf_features, metadata)

if __name__ == '__main__':
    main()
