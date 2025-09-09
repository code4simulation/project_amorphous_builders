# ==============================================================================
# Merged Python Script for project_amorphous_builders
#
# This script combines the code from the following modules for easier review:
# - src/main.py
# - src/utils/config.py
# - src/utils/rdf_utils.py
# - src/data/loader.py
# - src/data/preprocessing.py
# - src/data/dataset.py
# - src/model/graph_network.py
# - src/model/diffusion.py
# - src/model/train.py
# - src/model/generate.py
# ==============================================================================

import logging
import yaml
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
from ase.io import read, write
from ase.geometry import get_distances
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# --- SECTION: MAIN EXECUTION (from src/main.py) ---
# ==============================================================================
def main():
    config = load_config(sys.argv[1])
    mode = config.get('mode')

    if not config.validate():
        raise RuntimeError("Config validation failed")

    # using logging 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"사용 디바이스: {device}")

    if args.mode == 'preprocess':
        extxyz_path = config.get('data.extxyz_path')
        dataset_path = config.get('data.dataset_path')
        graph_cutoff = config.get('data.graph_cutoff', 5.0)
        r_max = config.get('data.r_max', 10.0)
        n_bins = config.get('data.n_bins', 100)
        normalize_rdf = config.get('data.normalize_rdf', True)

        structures = load_extxyz(extxyz_path)
        dataset = create_dataset_from_structures(
            structures,
            graph_cutoff=graph_cutoff,
            r_max=r_max,
            n_bins=n_bins,
            normalize_rdf=normalize_rdf
        )
        dataset.save(dataset_path)
        print(f"Preprocessing complete: saved to {dataset_path}")
        return

    elif mode == 'train':
        atoms_list = load_atomic_data(config['data']['train_path'])
        if not atoms_list: return
        
        preprocessed_data = preprocess_data(atoms_list, normalize=True)
        dataset = AtomicStructureDataset(preprocessed_data)
        
        train_model(config, diffusion_model, dataset)
        
    elif mode == 'generate':
        generate_structures(config, diffusion_model)

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

# --- from src/utils/rdf_utils.py ---
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

    dr_mic = torch.matmul(dfrac_wrapped, lattice)  # (A,B,3)

    pair_dists = torch.norm(dr_mic, dim=-1).flatten()  # (A*B,)
    pair_dists = pair_dists[pair_dists > 1e-8]  # 자기 자신 제외

    if pair_dists.numel() == 0:
        return torch.zeros_like(bin_centers_t, device=device)

    # Gaussian kernel 적용 (미분 가능)
    norm_factor = math.sqrt(2.0 * math.pi) * sigma
    diff_r = pair_dists.unsqueeze(1) - bin_centers_t.unsqueeze(0)  # (P, M)
    kernel = torch.exp(-0.5 * (diff_r / sigma) ** 2) / norm_factor

    prdf_unnorm = kernel.sum(dim=0)  # (M,)

    volume = float(atoms.get_volume())
    nA = len(idx1)
    nB = len(idx2)
    dr_val = float(step)
    r = bin_centers_t
    rho = nB / volume
    denom = (nA * rho * (4.0 * math.pi * (r ** 2) * dr_val)).to(device)
    denom = torch.where(denom == 0, torch.ones_like(denom, device=device), denom)
    prdf = prdf_unnorm / denom
    return prdf

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
            # Min-Max 정규화 [0, 1]
            rdf_min = np.min(rdf)
            rdf_max = np.max(rdf)
            if rdf_max - rdf_min > 1e-10:
                normalized = (rdf - rdf_min) / (rdf_max - rdf_min)
            else:
                normalized = np.zeros_like(rdf)
        elif method == 'standard':
            # 표준 정규화 (평균 0, 표준편차 1)
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

def batch_calculate_rdf(
    structures: List[Atoms], 
    r_max: float = 10.0, 
    n_bins: int = 200,
    normalize: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Atoms 구조 배치에 대해 RDF를 계산합니다.
    
    Args:
        structures (List[Atoms]): Atoms 객체 리스트
        r_max (float): 최대 거리 (Å)
        n_bins (int): 히스토그램 빈 개수
        normalize (bool): 정규화 적용 여부
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: (거리 배열, RDF 값 배열) 리스트
    """
    rdf_results = []
    for i, atoms in enumerate(structures):
        try:
            r, rdf = calculate_rdf(atoms, r_max, n_bins)
            if normalize:
                rdf = normalize_rdf(rdf)
            rdf_results.append((r, rdf))
            
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}개 구조 RDF 계산 완료")
                
        except Exception as e:
            logger.warning(f"구조 {i} RDF 계산 실패: {str(e)}")
            rdf_results.append((np.zeros(n_bins), np.zeros(n_bins)))
            continue
    
    logger.info(f"총 {len(rdf_results)}개 구조 RDF 계산 완료")
    return rdf_results

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
    normalize_rdf: bool = True
) -> AmorphousDataset:
    """
    ASE Atoms 객체 리스트로부터 데이터셋 생성
    
    Args:
        structures: ASE Atoms 객체 리스트
        graph_cutoff: 그래프 구성 시 cutoff distance (Å)
        r_max: RDF 계산 최대 거리 (Å)
        n_bins: RDF 히스토그램 bin 개수
        normalize_rdf: RDF 정규화 여부
        
    Returns:
        AmorphousDataset: 변환된 데이터셋
    """
    from .loader import atoms_to_graph
    from .preprocessing import calculate_rdf, normalize_rdf
    
    logger.info("데이터셋 생성 시작")
    
    graphs = []
    rdf_features = []
    
    for i, atoms in enumerate(structures):
        try:
            # 그래프 변환
            graph = atoms_to_graph(atoms, graph_cutoff=graph_cutoff)
            graphs.append(graph)
            
            # RDF 계산
            rdf = calculate_rdf(atoms, r_max=r_max, n_bins=n_bins)
            if normalize_rdf:
                rdf = normalize_rdf(rdf)
            rdf_features.append(rdf)
            
            if (i + 1) % 10 == 0:
                logger.info(f"{i + 1}/{len(structures)}개 구조 처리 완료")
                
        except Exception as e:
            logger.warning(f"구조 {i} 변환 실패: {str(e)}")
            continue
    
    # RDF 특징을 numpy 배열로 변환
    rdf_features = np.array(rdf_features) if rdf_features else None
    
    logger.info(f"총 {len(structures)}개 구조 중 {len(graphs)}개 성공적으로 변환")
    
    # 메타데이터 준비
    metadata = {
        'graph_cutoff': graph_cutoff,
        'r_max': r_max,
        'n_bins': n_bins,
        'normalize_rdf': normalize_rdf,
        'source_structures_count': len(structures),
        'converted_structures_count': len(graphs)
    }
    
    return AmorphousDataset(graphs, rdf_features, metadata)

# ==============================================================================
# --- SECTION: MODEL (from src/model/) ---
# ==============================================================================
# --- from src/model/graph_network.py ---
class EdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super(EdgeConv, self).__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),  # +1 for edge_attr (distance)
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        logger.debug(f"EdgeConv 초기화: in_channels={in_channels}, out_channels={out_channels}")
    
    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels], edge_index: [2, E], edge_attr: [E, 1]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: [E, in_channels], x_j: [E, in_channels], edge_attr: [E, 1]
        input = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 2*in_channels + 1]
        return self.mlp(input)

class ConditionalGraphNetwork(nn.Module):
    """조건부 그래프 신경망"""
    def __init__(
        self, 
        node_dim: int = 3,  # positions (x, y, z)
        edge_dim: int = 1,  # distance
        hidden_dim: int = 128,
        num_layers: int = 6,
        condition_dim: int = 100  # RDF feature dimension
    ):
        super(ConditionalGraphNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # 조건(RDF) 처리 MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 초기 노드 특징 변환
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # 에지 컨볼루션 레이어들
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(EdgeConv(hidden_dim, hidden_dim))
        
        # 시간 임베딩 (Diffusion steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 최종 출력 레이어
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)  # 노이즈 예측 (x, y, z)
        )
        
        logger.info(f"ConditionalGraphNetwork 초기화: "
                   f"node_dim={node_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, condition_dim={condition_dim}")
    
    def forward(
        self,
        x: Tensor,                     # [N, in_dim]
        edge_index: Tensor,            # [2, E]
        edge_attr: Tensor | None,      # [E, e_dim] or None
        t: Tensor,                     # [B] or [B, 1] (graph-level)
        batch: Tensor | None = None,   # [N], node->graph assignment in {0..B-1}
        condition: Tensor | None = None  # typically [B, c_dim], or [B], or [N, c_dim]
    ) -> Tensor:
        """
        Conditional graph network forward pass.

        Shapes
        ------
        N: total_nodes, B: batch_size (#graphs)
        x: [N, in_dim]
        t: [B] or [B,1]  --> time_features: [B, H]
        condition: [B, c_dim] or [B] or [N, c_dim] or None
        --> cond_features: [B, H] (then broadcast to nodes via [batch])
        batch: [N] with values in {0..B-1}

        Returns
        -------
        Tensor: [N, out_dim]
        """

        N = x.size(0)
        # 허용: 배치 비어있을 수 없음(일반 PyG DataLoader에서는 항상 존재).
        if batch is None:
            # 단일 그래프일 가능성(B=1)만 허용: t가 스칼라/길이1이면 batch를 0으로 채워 생성
            if t.numel() == 1:
                batch = torch.zeros(N, dtype=torch.long, device=x.device)
                B = 1
            else:
                raise ValueError("`batch` is required when B > 1 (multi-graph batches).")
        else:
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # 1) 노드 임베딩
        h = self.input_mlp(x)  # [N, H]
        H = h.size(1)

        # 2) 시간 임베딩 (그래프 단위 -> 노드로 broadcast)
        t = t.view(B, 1) if t.dim() == 1 else t  # [B,1]
        time_features = self.time_mlp(t.to(dtype=h.dtype))  # [B, H]
        assert time_features.size() == (B, H), f"time_features {time_features.shape} != {(B,H)}"

        # 3) 조건 임베딩 (여러 형태 허용)
        if condition is None:
            cond_features = torch.zeros_like(time_features)  # [B, H]
        else:
            if condition.dim() == 1 and condition.size(0) == B:
                # [B] -> [B,1]
                cond_in = condition.view(B, 1)
            elif condition.dim() == 2 and condition.size(0) == B:
                # [B, c_dim]
                cond_in = condition
            elif condition.size(0) == N:
                # [N, c_dim] (노드 단위 조건) -> 그래프 평균 등으로 축약
                try:
                    from torch_scatter import scatter_mean
                except ImportError as e:
                    raise ImportError(
                        "Node-level condition provided but torch_scatter is not installed. "
                        "Install torch-scatter or provide condition as [B, c_dim]."
                    ) from e
                cond_in = scatter_mean(condition, batch, dim=0, dim_size=B)  # [B, c_dim]
            else:
                raise ValueError(
                    f"Unsupported condition shape {tuple(condition.shape)}; "
                    f"expected [B], [B, c_dim], or [N, c_dim] with N={N}, B={B}."
                )

            cond_features = self.condition_mlp(cond_in)  # [B, H]

        assert cond_features.size() == (B, H), f"cond_features {cond_features.shape} != {(B,H)}"

        # 4) 그래프 단위 특징을 노드 단위로 브로드캐스트하여 결합
        h = h + time_features[batch] + cond_features[batch]  # [N,H]

        # 5) 메시지 패싱 with residual
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr)) + h  # [N,H]

        # 6) 출력 사상
        out = self.output_mlp(h)  # [N, out_dim]
        return out

# --- from src/model/diffusion.py ---
class DiffusionProcess:
    """Diffusion 프로세스 관리 클래스"""
    
    def __init__(
        self, 
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_timesteps: int = 1000,
        schedule: str = 'linear'
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # β_t 스케줄 설정
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            # Cosine schedule (Improved DDPM)
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # ᾱ_t = ∏_{s=1}^t α_s
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # √ᾱ_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # √(1 - ᾱ_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        logger.info(f"DiffusionProcess 초기화: "
                   f"timesteps={num_timesteps}, schedule={schedule}, "
                   f"beta_range=[{beta_start}, {beta_end}]")
    
    def forward_diffusion(self, x0, t, noise=None, batch=None):
        """
        x0: [total_nodes, 3]
        t: [batch_size] or [num_graphs] -- diffusion time indices
        batch: [total_nodes] -- node-to-graph batch indices
        """
        device = x0.device
        if noise is None:
            noise = torch.randn_like(x0)
        # --- 타입 강제 변환 ---
        t = t.long()
        batch = batch.long()
        # Broadcast t to each node in batch
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][batch].unsqueeze(-1)  # [total_nodes, 1]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][batch].unsqueeze(-1)
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        주어진 시간 스텝 t에 해당하는 값을 a에서 추출합니다.
        
        Args:
            a: 타임스텝별 값 [T]
            t: 시간 스텝 [B]
            x_shape: 출력 텐서의 형태
            
        Returns:
            t에 해당하는 값 [B, 1, 1, ...] (x_shape에 맞게 브로드캐스트 가능)
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionSampler:
    """
    Reverse diffusion sampling loop using DiffusionProcess + ConditionalGraphNetwork.
    """

    def __init__(self, process: DiffusionProcess, model: nn.Module, device: str = "cpu"):
        self.process = process
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        n_atoms: int,
        lattice_vector: np.ndarray,
        n_steps: Optional[int] = None,
    ) -> Tuple[list[str], np.ndarray]:
        """
        Generate atomic positions from noise.

        Args:
            cond: dict with keys {"rdf": ndarray, "formula": str, "n_atoms": int}
            n_atoms: number of atoms
            lattice_vector: (3,3) numpy array
            n_steps: number of diffusion steps (default: process.num_timesteps)

        Returns:
            elements: list[str]
            positions: (N,3) ndarray
        """
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from generate import parse_formula

        n_steps = n_steps or self.process.num_timesteps
        betas = self.process.betas.to(self.device)
        alphas = self.process.alphas.to(self.device)
        alphas_cumprod = self.process.alphas_cumprod.to(self.device)

        # 초기 노이즈 위치
        x_t = torch.randn(n_atoms, 3, device=self.device)

        # inside reverse loop after computing x_t (torch tensor)
        if target_rdf is not None and guidance_strength>0:
            x_t.requires_grad_(True)
            pred_g = differentiable_rdf(x_t, bins_t, sigma=some_sigma, lattice=lattice)
            rdf_loss = F.mse_loss(pred_g, target_rdf_t)
            grad = torch.autograd.grad(rdf_loss, x_t)[0]
            x_t = x_t - guidance_step_size * grad  # small step towards reducing RDF loss
            x_t = x_t.detach()

        # 조건 feature (RDF → torch tensor)
        if isinstance(cond["rdf"], np.ndarray):
            rdf_feat = torch.tensor(cond["rdf"], dtype=torch.float32, device=self.device)
        else:
            rdf_feat = torch.randn(100, device=self.device) # fallback

        elements = parse_formula(cond["formula"], n_atoms)

        batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        for t in reversed(range(n_steps)):
            t_tensor = torch.full((1,), t, dtype=torch.long, device=self.device)

            # GNN을 통한 노이즈 예측
            noise_pred = self.model(
                x=x_t,
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                edge_attr=None,
                t=t_tensor,
                batch=batch,
                condition=rdf_feat.unsqueeze(0), # [1, cond_dim]
            )

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # x_{t-1} 샘플링
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            x_t = coef1 * (x_t - coef2 * noise_pred) + torch.sqrt(beta_t) * noise

        return elements, x_t.cpu().numpy()

# ==============================================================================
# --- SECTION: TRAINING & GENERATION (from src/train.py, src/generate.py) ---
# ==============================================================================

# --- from src/train.py ---
class Trainer:  
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self.config.get('training.device')
        
        # 출력 디렉토리 설정
        self.output_dir = self.config.get('training.output_dir')
        self.device = self.config.get('training.device')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))
        
        # 모델 및 옵티마이저 초기화
        self._setup_model()
        
        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"Trainer 초기화 완료: device={self.device}")
    
    def _setup_model(self):
        from model.graph_network import ConditionalGraphNetwork
        from model.diffusion import DiffusionProcess
        
        # Diffusion 프로세스 초기화
        diffusion_config = self.config.get('diffusion')
        self.diffusion = DiffusionProcess(
            beta_start=float(diffusion_config['beta_start']),
            beta_end=float(diffusion_config['beta_end']),
            num_timesteps=diffusion_config['num_timesteps'],
            schedule=diffusion_config['schedule']
        )
        
        # 그래프 네트워크 초기화
        model_config = self.config.get('model')
        self.model = ConditionalGraphNetwork(
            node_dim=model_config['node_dim'],
            edge_dim=model_config['edge_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            condition_dim=model_config['condition_dim']
        ).to(self.device)
        
        # 옵티마이저 및 스케줄러 설정
        training_config = self.config.get('training')
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = float(training_config['learning_rate']),
            weight_decay= float(training_config['weight_decay'])
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_config['scheduler_step_size'],
            gamma=training_config['scheduler_gamma']
        )
        
        # 체크포인트 로드 (있는 경우)
        if training_config.get('resume_from_checkpoint'):
            self.load_checkpoint(training_config['resume_from_checkpoint'])
        
        logger.info(f"모델 설정 완료: 파라미터 수 {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, data in enumerate(train_loader):
            if not hasattr(data, 'pos') or data.pos is None:
                logger.error(f"배치 {batch_idx}에서 pos가 None입니다. 데이터셋을 점검하세요.")
                continue  # 또는 raise ValueError
            
            data = data.to(self.device)
            
            t = torch.randint(0, self.diffusion.num_timesteps, (data.num_graphs,), device=data.pos.device)
            t = t.long()  # 혹시 float로 변환된 적 있다면 long으로 보장
            noisy_positions, noise = self.diffusion.forward_diffusion(
                data.pos, t, noise=None, batch=data.batch
            )

            self.optimizer.zero_grad()
            predicted_noise = self.model(
                x=data.x if hasattr(data, "x") else data.pos,   # 데이터 구조에 맞게
                edge_index=data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                t=t,                           # [B] 또는 [B,1]
                batch=data.batch,        # [N]   반드시 추가
                condition=getattr(data, "condition", None)
            )
            
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.get('training.log_interval') == 0:
                logger.info(f"Epoch {self.current_epoch} [{batch_idx}/{num_batches}] "
                           f"Loss: {loss.item():.6f}")
                
                step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                t = torch.randint(
                    0, self.diffusion.num_timesteps, (data.num_graphs,), 
                    device=self.device
                ).long()
                
                noisy_positions, noise = self.diffusion.forward_diffusion(
                    data.pos, t, noise=None
                )

                predicted_noise = self.model(
                    x=noisy_positions,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    t=t,
                    condition=data.rdf
                )

                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.output_dir, 
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"새로운 최고 성능 모델 저장: loss={self.best_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"체크포인트 로드 완료: {checkpoint_path}, 에폭 {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """전체 학습 프로세스"""
        training_config = self.config.get('training')
        num_epochs = training_config['num_epochs']
        
        logger.info(f"학습 시작: 총 {num_epochs} 에폭")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} 완료 - Train Loss: {train_loss:.6f}")
            
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.6f}")
                self.writer.add_scalar('val/loss', val_loss, epoch)
            
            self.scheduler.step()
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % training_config['checkpoint_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        logger.info("학습 완료!")
        self.writer.close()

# --- from src/generate.py ---
"""Generate atomic structures sacrifying target partial RDF using trained diffusion model"""
"""Build up code def or class"""
def generate_structures(config: Dict, diffusion_model: DiffusionModel) -> List[Atoms]:
    """ code... """
    return generated_atoms_list

if __name__ == '__main__':
    main()
