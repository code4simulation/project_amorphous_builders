import logging
import torch
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
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor,
        t: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 노드 특징 [N, node_dim] (노이즈가 추가된 positions)
            edge_index: 에지 인덱스 [2, E]
            edge_attr: 에지 속성 [E, edge_dim] (거리)
            t: 시간 스텝 [1] 또는 [B] (배치인 경우)
            condition: 조건(RDF) 특징 [condition_dim]
            
        Returns:
            예측된 노이즈 [N, node_dim]
        """

        if t.dtype != torch.float32:
            t = t.float()

        # 조건 특징 인코딩
        condition_features = self.condition_mlp(condition)  # [hidden_dim]
        
        # 시간 특징 인코딩
        time_features = self.time_mlp(t.view(-1, 1))  # [batch_size, hidden_dim]
        
        # 노드 특징 인코딩
        h = self.node_encoder(x)  # [N, hidden_dim]
        
        # 조건과 시간 특징을 모든 노드에 브로드캐스트하여 추가
        h = h + condition_features.unsqueeze(0) + time_features.unsqueeze(0)
        
        # 그래프 컨볼루션 레이어 적용
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr)) + h  # Residual connection
        
        # 최종 출력
        return self.output_mlp(h)
