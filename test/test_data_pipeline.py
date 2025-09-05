import logging
import sys
import os

# src 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data.loader import load_extxyz, batch_structures_to_graphs
from data.preprocessing import batch_calculate_rdf, normalize_rdf
from data.dataset import create_dataset_from_structures, AmorphousDataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_data_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """데이터 처리 파이프라인 테스트"""
    logger.info("데이터 처리 파이프라인 테스트 시작")
    
    # 1. extxyz 파일 로드
    logger.info("1. extxyz 파일 로드 중...")
    try:
        structures = load_extxyz('test_structures.extxyz')
        logger.info(f"로드된 구조 개수: {len(structures)}")
        logger.info(f"첫 번째 구조 정보: {len(structures[0])}개 원자")
        logger.info(f"원자 종류: {set(structures[0].get_chemical_symbols())}")
    except Exception as e:
        logger.error(f"파일 로드 실패: {e}")
        return False
    
    # 2. 그래프 변환
    logger.info("2. 그래프 변환 중...")
    try:
        graphs = batch_structures_to_graphs(structures, graph_cutoff=5.0)
        logger.info(f"변환된 그래프 개수: {len(graphs)}")
        logger.info(f"첫 번째 그래프 정보: {graphs[0]}")
        logger.info(f"노드 수: {graphs[0].num_nodes}, 에지 수: {graphs[0].edge_index.shape[1]}")
    except Exception as e:
        logger.error(f"그래프 변환 실패: {e}")
        return False
    
    # 3. RDF 계산
    logger.info("3. RDF 계산 중...")
    try:
        rdf_data = batch_calculate_rdf(structures, r_max=10.0, n_bins=100, normalize=True)
        logger.info(f"계산된 RDF 개수: {len(rdf_data)}")
        logger.info(f"첫 번째 RDF 모양: 거리={rdf_data[0][0].shape}, 값={rdf_data[0][1].shape}")
        
        # RDF 시각화 (선택사항)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i, (r, rdf) in enumerate(rdf_data[:3]):  # 처음 3개만 시각화
            plt.plot(r, rdf, label=f"구조 {i+1}")
        plt.xlabel('거리 (Å)')
        plt.ylabel('RDF')
        plt.title('Radial Distribution Function')
        plt.legend()
        plt.savefig('test_rdf_plot.png')
        logger.info("RDF 플롯 저장 완료: test_rdf_plot.png")
        
    except Exception as e:
        logger.error(f"RDF 계산 실패: {e}")
        return False
    
    # 4. 데이터셋 생성
    logger.info("4. 데이터셋 생성 중...")
    try:
        dataset = create_dataset_from_structures(
            structures, 
            graph_cutoff=5.0,
            r_max=10.0,
            n_bins=100,
            normalize_rdf=True
        )
        logger.info(f"데이터셋 크기: {len(dataset)}")
        
        # 데이터셋 통계
        stats = dataset.get_rdf_stats()
        logger.info(f"RDF 통계: {stats}")
        
        # 샘플 데이터 확인
        sample = dataset[0]
        logger.info(f"샘플 데이터: 그래프={type(sample)}, RDF 모양={sample.rdf.shape}")
        
    except Exception as e:
        logger.error(f"데이터셋 생성 실패: {e}")
        return False
    
    logger.info("데이터 처리 파이프라인 테스트 완료!")
    return True

if __name__ == "__main__":
    success = test_data_pipeline()
    if success:
        print("모든 테스트가 성공적으로 완료되었습니다!")
        print("로그 파일: test_data_pipeline.log")
        print("RDF 시각화: test_rdf_plot.png")
    else:
        print("테스트 중 오류가 발생했습니다. 로그 파일을 확인하세요.")
        sys.exit(1)
