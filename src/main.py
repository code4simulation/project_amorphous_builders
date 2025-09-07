import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
from datetime import datetime
from typing import Dict, Any, Tuple

from data.loader import load_extxyz, batch_structures_to_graphs
from data.preprocessing import batch_calculate_rdf, normalize_rdf
from data.dataset import create_dataset_from_structures, AmorphousDataset
from utils.config import load_config, Config

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
                batch=data.batch,        # [N]  👈 반드시 추가
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
        """체크포인트 저장"""
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
        training_config = self.config.get('training')
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

def main():
    """메인 학습 함수"""
    # 설정 파일 로드
    config = load_config('input.yaml')
    mode = config.get('mode')

    if not config.validate():
        raise RuntimeError("Config validation failed")
    
    if mode == 'preprocess':
        extxyz_path = config.get('data.extxyz_path')
        dataset_path = config.get('data.dataset_path')
        graph_cutoff = config.get('data.graph_cutoff', 5.0)
        r_max = config.get('data.r_max', 10.0)
        n_bins = config.get('data.n_bins', 100)
        normalize_rdf = config.get('data.normalize_rdf', True)

        # 1. extxyz 파일 로드
        structures = load_extxyz(extxyz_path)
        # 2. 그래프/특징 추출
        dataset = create_dataset_from_structures(
            structures,
            graph_cutoff=graph_cutoff,
            r_max=r_max,
            n_bins=n_bins,
            normalize_rdf=normalize_rdf
        )
        # 3. pt 파일로 저장
        dataset.save(dataset_path)
        print(f"Preprocessing complete: saved to {dataset_path}")
        return

    elif mode == 'train':
        from data.dataset import AmorphousDataset
        from torch_geometric.loader import DataLoader

        dataset = AmorphousDataset.load(config.get('data.dataset_path'))
        train_loader = DataLoader(
        dataset, 
        batch_size=config.get('training.batch_size'),
        shuffle=False,
        #num_workers=int(config.get('training.num_workers'))
        )
     
        # 검증 데이터 로더 (있는 경우)
        val_loader = None
        if 'val_dataset_path' in config.get('data'):
            val_dataset = AmorphousDataset.load(config.get('data.val_dataset_path'))
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('training.batch_size'),
                shuffle=False,
            )

        # 학습기 생성 및 학습 실행
        trainer = Trainer(config)
        trainer.train(train_loader, val_loader)

    elif mode == 'generate':
        dataset_path = config.get('data.dataset_path')
        dataset = AmorphousDataset.load(dataset_path)
        # ...이하 생성 코드...

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
