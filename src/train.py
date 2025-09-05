import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
from datetime import datetime
from typing import Dict, Any, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    """Diffusion 모델 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        # 출력 디렉토리 설정
        self.output_dir = config['training']['output_dir']
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
        """모델 및 옵티마이저 설정"""
        from model.graph_network import ConditionalGraphNetwork
        from model.diffusion import DiffusionProcess
        
        # Diffusion 프로세스 초기화
        diffusion_config = self.config['diffusion']
        self.diffusion = DiffusionProcess(
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            num_timesteps=diffusion_config['num_timesteps'],
            schedule=diffusion_config['schedule']
        )
        
        # 그래프 네트워크 초기화
        model_config = self.config['model']
        self.model = ConditionalGraphNetwork(
            node_dim=model_config['node_dim'],
            edge_dim=model_config['edge_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            condition_dim=model_config['condition_dim']
        ).to(self.device)
        
        # 옵티마이저 및 스케줄러 설정
        training_config = self.config['training']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
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
        """한 에폭 동안 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, data in enumerate(train_loader):
            # 데이터를 device로 이동
            data = data.to(self.device)
            
            # 랜덤 타임스텝 샘플링
            t = torch.randint(
                0, self.diffusion.num_timesteps, (data.num_graphs,), 
                device=self.device
            ).long()
            
            # Forward diffusion (노이즈 추가)
            noisy_positions, noise = self.diffusion.forward_diffusion(
                data.pos, t, noise=None
            )
            
            # 모델 예측
            self.optimizer.zero_grad()
            predicted_noise = self.model(
                x=noisy_positions,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                t=t,
                condition=data.rdf
            )
            
            # 손실 계산 (MSE between predicted and actual noise)
            loss = F.mse_loss(predicted_noise, noise)
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            
            if batch_idx % self.config['training']['log_interval'] == 0:
                logger.info(f"Epoch {self.current_epoch} [{batch_idx}/{num_batches}] "
                           f"Loss: {loss.item():.6f}")
                
                # TensorBoard에 기록
                step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """검증 데이터로 모델 평가"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                # 랜덤 타임스텝 샘플링
                t = torch.randint(
                    0, self.diffusion.num_timesteps, (data.num_graphs,), 
                    device=self.device
                ).long()
                
                # Forward diffusion
                noisy_positions, noise = self.diffusion.forward_diffusion(
                    data.pos, t, noise=None
                )
                
                # 모델 예측
                predicted_noise = self.model(
                    x=noisy_positions,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    t=t,
                    condition=data.rdf
                )
                
                # 손실 계산
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 일반 체크포인트
        checkpoint_path = os.path.join(
            self.output_dir, 
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"새로운 최고 성능 모델 저장: loss={self.best_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
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
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        
        logger.info(f"학습 시작: 총 {num_epochs} 에폭")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} 완료 - Train Loss: {train_loss:.6f}")
            
            # 검증 (있는 경우)
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.6f}")
                self.writer.add_scalar('val/loss', val_loss, epoch)
            
            # 학습률 스케줄링
            self.scheduler.step()
            
            # 체크포인트 저장
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % training_config['checkpoint_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        logger.info("학습 완료!")
        self.writer.close()

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """메인 학습 함수"""
    # 설정 파일 로드
    config = load_config('input.yaml')
    
    # 데이터셋 로드
    from data.dataset import AmorphousDataset
    from torch_geometric.loader import DataLoader
    
    dataset = AmorphousDataset.load(config['data']['dataset_path'])
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # 검증 데이터 로더 (있는 경우)
    val_loader = None
    if 'val_dataset_path' in config['data']:
        val_dataset = AmorphousDataset.load(config['data']['val_dataset_path'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )
    
    # 학습기 생성 및 학습 실행
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
