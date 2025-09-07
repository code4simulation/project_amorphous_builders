import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
