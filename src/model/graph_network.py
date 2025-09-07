import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_network.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
