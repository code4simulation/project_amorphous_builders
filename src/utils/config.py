import yaml
import logging
from typing import Dict, Any, Optional
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """YAML 설정 파일을 파싱하고 관리하는 클래스"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        try:
            # utf-8 우선, 실패시 cp949 fallback
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
        """설정 값 조회 (점 표기법 지원: 'data.train_data_path')"""
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
        """필수 설정 값 검증"""
        required_keys = [
            'mode',
            'data.train_data_path',
            'model.node_dim',
            'model.hidden_dim',
            'training.device',
            'training.batch_size',
            'training.num_epochs'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"필수 설정 값이 누락되었습니다: {missing_keys}")
            return False
        
        # 모드별 추가 검증
        mode = self.get('mode')
        if mode == 'train':
            # 학습 모드 검증
            pass
        elif mode == 'generate':
            # 생성 모드 검증
            gen_required = [
                'generation.lattice_vector',
                'generation.density',
                'generation.chemical_formula',
                'generation.num_atoms',
                'generation.target_rdf'
            ]
            for key in gen_required:
                if self.get(key) is None:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.error(f"생성 모드 필수 설정 값이 누락되었습니다: {missing_keys}")
                return False
        
        logger.info("설정 검증 완료")
        return True
    
    def update(self, key: str, value: Any):
        """설정 값 업데이트"""
        keys = key.split('.')
        config_dict = self.config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
        logger.debug(f"설정 업데이트: {key} = {value}")
    
    def save(self, path: Optional[str] = None):
        """현재 설정을 YAML 파일로 저장"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"설정 파일 저장 완료: {save_path}")
    
    def __str__(self) -> str:
        """설정 내용 문자열 표현"""
        return yaml.dump(self.config, default_flow_style=False)

def load_config(config_path: str) -> Config:
    """설정 파일 로드 헬퍼 함수"""
    return Config(config_path)
