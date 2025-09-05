import logging
import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from typing import List, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
