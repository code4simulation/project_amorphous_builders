import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from ase import Atoms
from torch_geometric.data import Data

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AmorphousDataset(Dataset):
    """
    비정질 구조 데이터셋 클래스
    """
    
    def __init__(
        self, 
        graphs: List[Data],
        rdf_data: List[Tuple[np.ndarray, np.ndarray]],
        transform: Optional[callable] = None
    ):
        """
        Args:
            graphs (List[Data]): 그래프 데이터 리스트
            rdf_data (List[Tuple[np.ndarray, np.ndarray]]): RDF 데이터 리스트
            transform (callable, optional): 데이터 변환 함수
        """
        self.graphs = graphs
        self.rdf_data = rdf_data
        self.transform = transform
        
        # 데이터 검증
        self._validate_data()
        logger.info(f"데이터셋 초기화 완료: {len(self)}개 샘플")
    
    def _validate_data(self):
        """데이터 무결성 검증"""
        if len(self.graphs) != len(self.rdf_data):
            logger.error("그래프와 RDF 데이터의 길이가 일치하지 않습니다.")
            raise ValueError("그래프와 RDF 데이터의 길이가 일치하지 않습니다.")
        
        # RDF 데이터 형식 검증
        for i, (r, rdf) in enumerate(self.rdf_data):
            if len(r) != len(rdf):
                logger.error(f"인덱스 {i}: 거리 배열과 RDF 배열의 길이가 일치하지 않습니다.")
                raise ValueError(f"인덱스 {i}: 거리 배열과 RDF 배열의 길이가 일치하지 않습니다.")
        
        logger.info("데이터 검증 완료")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        try:
            graph = self.graphs[idx]
            r, rdf = self.rdf_data[idx]
            
            # RDF를 텐서로 변환
            rdf_tensor = torch.tensor(rdf, dtype=torch.float)
            
            # 그래프에 RDF 정보 추가
            graph.rdf = rdf_tensor
            
            if self.transform:
                graph = self.transform(graph)
                
            return graph
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생 (인덱스 {idx}): {str(e)}")
            raise
    
    def get_rdf_stats(self) -> dict:
        """
        RDF 통계 정보를 계산합니다.
        
        Returns:
            dict: RDF 통계 정보
        """
        all_rdf = np.concatenate([rdf for _, rdf in self.rdf_data])
        
        stats = {
            'mean': np.mean(all_rdf),
            'std': np.std(all_rdf),
            'min': np.min(all_rdf),
            'max': np.max(all_rdf),
            'median': np.median(all_rdf)
        }
        
        logger.info(f"RDF 통계 계산 완료: 평균={stats['mean']:.3f}, 표준편차={stats['std']:.3f}")
        return stats

def create_dataset_from_structures(
    structures: List[Atoms],
    graph_cutoff: float = 5.0,
    r_max: float = 10.0,
    n_bins: int = 200,
    normalize_rdf: bool = True
) -> AmorphousDataset:
    """
    Atoms 구조 리스트로부터 데이터셋을 생성합니다.
    
    Args:
        structures (List[Atoms]): Atoms 객체 리스트
        graph_cutoff (float): 그래프 구성 시 cutoff distance (Å)
        r_max (float): RDF 계산 최대 거리 (Å)
        n_bins (int): RDF 히스토그램 빈 개수
        normalize_rdf (bool): RDF 정규화 적용 여부
        
    Returns:
        AmorphousDataset: 생성된 데이터셋
    """
    from .loader import batch_structures_to_graphs
    from .preprocessing import batch_calculate_rdf
    
    logger.info("데이터셋 생성 시작")
    
    # 구조를 그래프로 변환
    graphs = batch_structures_to_graphs(structures, graph_cutoff)
    
    # RDF 계산
    rdf_data = batch_calculate_rdf(structures, r_max, n_bins, normalize_rdf)
    
    # 데이터셋 생성
    dataset = AmorphousDataset(graphs, rdf_data)
    
    logger.info("데이터셋 생성 완료")
    return dataset
