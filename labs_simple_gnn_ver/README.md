# 프로젝트 요약: 확산 모델을 이용한 비정질 원자구조 생성

## 1. 프로젝트 목표

**핵심 목표:** 조건부 확산 모델(Conditional Diffusion Model)을 사용하여 물리적으로 타당한 **비정질(amorphous) 원자 구조**를 생성하는 프로그램을 개발한다.

- **입력 데이터:** 분자동역학(MD) 시뮬레이션으로 생성된 `.extxyz` 포맷의 원자 구조 데이터.
- **주요 기능:**
    1.  주어진 조건(원자 밀도, 화학식, 격자 벡터)에 맞는 새로운 구조를 생성한다.
    2.  생성 과정에서 **가변적인 원자 수**를 유연하게 처리할 수 있어야 한다.
    3.  물리적 제약 조건으로 **부분 방사상 분포 함수(Partial RDF)**를 활용하며, 이는 미분 가능한(differentiable) 형태로 모델에 통합된다.

## 2. 기술적 요구사항

- **데이터 표현:** 원자 구조를 그래프(Graph) 형태로 표현한다. (원자=노드, 원자 간 거리=간선)
- **모델 아키텍처:** 그래프 신경망(GNN)을 기반으로 한 확산 모델을 사용한다.
- **RDF 처리:**
    - `ASE` 라이브러리를 활용하여 주기성을 고려한 Partial RDF를 계산한다.
    - Gaussian Kernel을 적용하여 RDF를 부드럽게 만들어 미분 가능하도록 처리한다.
    - 원자쌍(e.g., Si-O)은 순서에 무관하도록 원자 번호 기준 오름차순으로 정렬하여 키로 사용한다.
- **학습 파이프라인:**
    - 학습/검증 데이터셋 분리.
    - 조기 중단(Early Stopping), 최적 모델 저장, 주기적 체크포인트 저장 기능 구현.
    - Epoch별 학습 및 검증 손실(Loss) 로그 출력.
- **개발 환경:** Python, PyTorch, PyTorch Geometric, ASE, dscribe.

## 3. 방법론의 발전 과정 (Evolution)

### Phase 1: ACSF (Atom-Centered Symmetry Functions) 기반 접근

- **초기 아이디어:** 각 원자의 국소 환경을 `dscribe`의 ACSF를 이용해 고정된 크기의 특징 벡터로 변환하여 학습 데이터로 사용한다.
- **문제점 식별:** ACSF는 각 원자에 대해 고정된 차원의 벡터를 생성하므로, 전체 구조의 원자 수가 달라지면 입력 텐서의 크기가 변한다. 이는 일반적인 딥러닝 모델에서 배치(batch) 처리를 복잡하게 만들며, 가변 원자 수를 처리하기 위해 복잡한 패딩(padding) 및 마스킹(masking) 기법이 필수로 요구된다. 이는 모델의 유연성과 확장성을 저해하는 근본적인 한계로 지적되었다.

### Phase 2: GNN (Graph Neural Network) 기반으로 전환

- **개선 아이디어:** ACSF의 한계를 극복하기 위해, 원자 구조를 그래프로 표현하는 방식으로 전환한다.
    - **원자:** 노드(Node)
    - **원자 간 상호작용 (거리 임계값 이내):** 간선(Edge)
- **GNN의 장점:**
    1.  **가변 크기 처리:** 그래프의 노드와 간선 수가 달라도 자연스럽게 처리할 수 있어, 가변 원자 수 문제에 완벽히 대응한다.
    2.  **순서 불변성(Permutation Invariance):** 원자 인덱스 순서가 바뀌어도 동일한 그래프로 인식하여 구조적 특징을 일관성 있게 학습한다.
    3.  **관계성 학습:** 메시지 패싱(Message Passing)을 통해 원자 간의 상호작용 및 전체 구조적 맥락을 직접 학습할 수 있다.
- **최종 결정:** 프로젝트의 핵심 요구사항인 '가변 원자 수 처리'를 위해 GNN 기반 아키텍처를 채택한다.

## 4. 최종 제안 아키텍처 (GNN 기반)

### 4.1. 데이터 전처리 및 데이터셋

- `.extxyz` 파일을 읽어 `torch_geometric.data.Data` 객체로 변환한다.
- 각 `Data` 객체는 다음 정보를 포함한다:
    - `x`: 노드 특징 벡터 (e.g., 원자 종류를 나타내는 원-핫 인코딩 또는 임베딩 인덱스).
    - `pos`: 원자의 3D 좌표 `(N, 3)`.
    - `edge_index`: 거리 임계값(e.g., 6.0 Å) 내에 있는 원자 쌍을 연결하는 간선 정보 `(2, num_edges)`.
    - `edge_attr` (선택): 간선의 속성 (e.g., 원자 간 거리).

```
# 파일: dataset.py
import torch
from torch_geometric.data import Data, Dataset
from ase.io import read

class GraphDataset(Dataset):
    def __init__(self, root_dir, max_radius=6.0):
        super().__init__()
        # ... 초기화 로직 ...

    def len(self):
        # ... 전체 파일 수 반환 ...

    def get(self, idx):
        atoms = read(self.files[idx], format='extxyz')
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        
        # 원자 종류를 임베딩 인덱스로 변환
        node_feats = ... 

        # 거리 기반 간선 정보(edge_index) 생성
        edge_index = ...

        return Data(x=node_feats, pos=pos, edge_index=edge_index)
```

### 4.2. 모델 아키텍처

- **GNN 모델:** `torch_geometric`의 GNN 레이어(e.g., `GCNConv`, `GINConv`)를 사용한 메시지 패싱 네트워크.
- **입력:** `Data` 객체 (그래프 데이터).
- **출력:** 각 노드(원자)에 대한 업데이트된 특징 벡터(임베딩). 이 벡터가 확산 과정의 대상이 된다.

```
# 파일: model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_node_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 3) # 예시: 좌표를 직접 예측

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        x = self.embedding(x.squeeze(-1))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 출력은 모델의 목적에 따라 달라짐 (e.g., 노이즈 예측, 좌표 예측 등)
        predicted_noise_on_pos = self.lin(x)
        return predicted_noise_on_pos
```

### 4.3. 학습 및 생성 프로세스

- **학습:**
    1.  `GraphDataset`과 `torch_geometric.loader.DataLoader`를 사용하여 가변 크기 그래프 배치를 효율적으로 구성한다.
    2.  확산 모델의 정방향 과정(Forward Process): 원본 원자 좌표(`pos`)에 점진적으로 노이즈를 추가한다.
    3.  역방향 과정(Reverse Process): GNN 모델은 노이즈가 낀 좌표와 그래프 구조를 입력받아, 추가된 노이즈를 예측하도록 학습한다.
    4.  손실 함수는 실제 노이즈와 예측된 노이즈 간의 차이(e.g., MSE)로 계산한다.
    5.  조건(밀도, 화학식 등)은 그래프의 전역 특징(global feature)으로 추가하거나 각 노드 특징에 결합하여 모델에 전달한다.

- **생성:**
    1.  생성할 원자 수만큼의 노드를 가진 그래프를 가정하고, 표준 정규 분포에서 초기 좌표를 샘플링한다.
    2.  주어진 조건과 함께, 학습된 GNN 모델을 사용하여 여러 스텝에 걸쳐 점진적으로 노이즈를 제거한다.
    3.  최종적으로 노이즈가 제거된 원자 좌표를 얻어 새로운 구조를 완성한다.

## 5. 핵심 유틸리티 코드

### 미분 가능한 RDF 계산

```
# 파일: utils.py
from ase.geometry import get_distances

def calculate_differentiable_rdf(atoms, r_max, n_bins, elements=None, sigma=0.1):
    # 1. ASE의 get_distances로 원자 쌍 거리 계산
    # 2. 히스토그램 대신, 각 거리에 대해 Gaussian 분포를 합산하여 부드러운 곡선 생성
    # 3. 물리적 의미에 맞게 정규화
    ...
    return r_values, g_r_smooth
```

## 6. 결론 및 다음 단계

- **결론:** 비정질 구조 생성을 위해 ACSF의 한계를 인지하고, 가변 원자 수 처리에 강점을 가진 **GNN 기반 조건부 확산 모델**로 아키텍처를 최종 결정했다.
- **다음 단계:**
    1.  `GraphDataset` 클래스를 완성하여 `.extxyz` 파일을 그래프 데이터로 변환하는 파이프라인을 구축한다.
    2.  GNN 기반 확산 모델의 상세 구조와 조건부 입력 처리 방식을 구체화한다.
    3.  그래프 데이터(특히 좌표 `pos`)에 대한 확산 및 역확산 프로세스(스케줄러 포함)를 구현한다.
    4.  전체 학습 및 생성 코드를 통합하여 실제 데이터로 모델을 훈련하고 성능을 검증한다.
```
