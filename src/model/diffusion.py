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
