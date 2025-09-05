import logging
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import os

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
    """비정질 구조 데이터셋 클래스"""
    
    def __init__(
        self, 
        graphs: List[Data],
        rdf_features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.graphs = graphs
        self.rdf_features = rdf_features
        self.metadata = metadata or {}
        
        # 데이터 검증
        self._validate_data()
        
        logger.info(f"데이터셋 초기화 완료: {len(self)}개 샘플")
    
    def _validate_data(self):
        """데이터 무결성 검증"""
        if len(self.graphs) == 0:
            raise ValueError("빈 그래프 리스트")
            
        if self.rdf_features is not None and len(self.graphs) != len(self.rdf_features):
            raise ValueError("그래프와 RDF 특징의 개수가 일치하지 않습니다")
        
        logger.info("데이터 검증 완료")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """인덱스에 해당하는 샘플 반환"""
        graph = self.graphs[idx]
        
        sample = {
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr,
            'pos': graph.pos,
            'num_nodes': graph.num_nodes,
            'atomic_numbers': getattr(graph, 'atomic_numbers', None)
        }
        
        # RDF 특징 추가 (있는 경우)
        if self.rdf_features is not None:
            sample['rdf'] = torch.tensor(self.rdf_features[idx], dtype=torch.float)
        
        return sample
    
    def save(self, path: str):
        """데이터셋을 파일로 저장"""
        try:
            # 저장할 데이터 준비
            save_data = {
                'graphs': self.graphs,
                'rdf_features': self.rdf_features,
                'metadata': self.metadata
            }
            
            # 디렉토리 생성 (없는 경우)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # 데이터 저장
            torch.save(save_data, path)
            logger.info(f"데이터셋 저장 완료: {path}")
            
        except Exception as e:
            logger.error(f"데이터셋 저장 중 오류 발생: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'AmorphousDataset':
        """파일에서 데이터셋 로드"""
        try:
            # 데이터 로드
            save_data = torch.load(path)
            
            # 데이터셋 인스턴스 생성
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
    
    def get_stats(self) -> Dict[str, Any]:
        """데이터셋 통계 정보 반환"""
        num_nodes = [graph.num_nodes for graph in self.graphs]
        num_edges = [graph.edge_index.shape[1] for graph in self.graphs]
        
        stats = {
            'num_samples': len(self),
            'avg_nodes': np.mean(num_nodes),
            'min_nodes': np.min(num_nodes),
            'max_nodes': np.max(num_nodes),
            'avg_edges': np.mean(num_edges),
            'min_edges': np.min(num_edges),
            'max_edges': np.max(num_edges),
        }
        
        # RDF 통계 (있는 경우)
        if self.rdf_features is not None:
            stats['rdf_shape'] = self.rdf_features.shape
            stats['rdf_mean'] = np.mean(self.rdf_features)
            stats['rdf_std'] = np.std(self.rdf_features)
        
        return stats

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
