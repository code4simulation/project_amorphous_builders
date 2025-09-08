# project_amorphous_builders

## Schematic Code Architecture

```
project_amorphous_builders/
│
├── examples/
│   └── ...           # 사용 예시, 샘플 config, 실험/테스트 케이스
│
├── src/
│   ├── __init__.py   # src 패키지 초기화
│   ├── main.py       # 주요 실행 진입점: config, 데이터 로드, 모델 초기화/학습/생성 전체 파이프라인 제공
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # extxyz 등 원자 구조 데이터 파일 로드. numpy/pandas/torch 타입 변환. 반환: 구조 배열/좌표/원자종 등
│   │   ├── preprocessing.py  # 데이터 전처리: 정규화, 증강, 마스킹 등 모델 입력 전 가공
│   │   └── dataset.py        # PyTorch Dataset/Loader 클래스 정의. 학습/테스트 데이터 관리, 배치 생성
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── graph_network.py  # GNN(U-Net, Graph Attention 등) 모델 정의. 입력: 구조, 그래프, timestep, noise 등
│   │   └── diffusion.py      # Diffusion process 코드. Forward/reverse step 함수, 노이즈 추가/제거, 샘플 생성 구현
│   │
│   ├── train.py      # 학습 루프 제어. 데이터셋, 모델, 옵티마이저, 로스 함수 초기화. 에폭/스텝 반복, 체크포인트 저장
│   ├── generate.py   # 학습된 모델 불러와서 새로운 구조 생성. Diffusion 샘플링, 결과 저장 및 시각화
│   │
│   └── utils/
│       ├── config.py         # yaml/json 설정 파일 파싱. 경로/파일명 오류 방지
│       ├── rdf_utils.py      # 
│       └── visualization.py  # 원자 구조 plot, 학습 곡선, 생성 결과 등 시각화 함수. matplotlib, plotly 사용 가능
│
├── test/
│   └── ...           # 유닛/통합 테스트 케이스
│
├── AI_CODING_GUIDELINES.md  # AI 코딩 가이드라인 및 아키텍처 설명
└── README.md                # 프로젝트/코드 구조 설명 문서
```

---

## 디렉토리 및 파일별 주요 기능 설명

- **examples/**: 샘플 config, 실험/테스트 예시를 제공. 파이프라인 동작 방법, 입력/출력 포맷 등 참조.
- **src/main.py**: 프로젝트 전체 파이프라인 실행 진입점. 설정 로드(config), 데이터 준비(loader, preprocessing, dataset), 모델 초기화(graph_network, diffusion), 학습(train), 샘플 생성(generate) 등 단계별 모듈 연결.
- **src/data/loader.py**: extxyz 등 원자 구조 파일을 파싱해 numpy, pandas, torch 형식 데이터로 변환. 구조 배열, 좌표, 원자종 등 반환. 이후 preprocessing으로 넘김.
- **src/data/preprocessing.py**: raw 데이터를 정규화, 증강, 마스킹 등 모델 입력에 적합하게 가공. normalize(data), augment(data) 등 함수 제공.
- **src/data/dataset.py**: PyTorch Dataset 클래스로 가공된 데이터를 배치 단위로 관리. 학습/생성 단계에서 iterator로 반복 사용.
- **src/model/graph_network.py**: GNN(U-Net, Graph Attention 등) 모델 정의. forward(x, edge_index, t) 등 함수 일관성 유지. 입력: 구조 정보, 그래프, timestep, noise 등.
- **src/model/diffusion.py**: Diffusion process 구현. forward_step, reverse_step 함수로 노이즈 추가/제거. 샘플 구조 생성. 모델의 forward, 연결 함수 명칭 일관성 필수.
- **src/train.py**: 학습 루프 제어. 데이터셋, 모델, 옵티마이저, 로스 함수 등 초기화 후 반복. 에폭/스텝 반복 및 체크포인트 저장. 데이터, 모델, config 등 입력 전달.
- **src/generate.py**: 학습된 모델을 사용해 새로운 구조 생성. Diffusion 샘플링, 결과 저장, 시각화 등 처리. 모델, 샘플링 파라미터, 저장 경로 등 인자 전달.
- **src/utils/config.py**: yaml/json 등 설정 파일 파싱. 경로/파일명 오류 방지, key와 코드 내 변수명 일치 주의.
- **src/utils/visualization.py**: 원자 구조 plot, 학습 곡선, 샘플 생성 결과 등 시각화 함수. matplotlib, plotly 등 지원. 결과 저장 경로/파일명 관리.
- **test/**: 각 모듈별 테스트 케이스. 데이터 입출력, 모델 forward, 학습 루프 등 검증.
- **AI_CODING_GUIDELINES.md**: AI 기반 코드 작성 시 일관성, 네이밍, 아키텍처, 주요 함수/클래스 명칭, 프롬프트 작성 규칙 등 가이드.
- **README.md**: 전체 프로젝트 및 코드 구조 설명, 각 디렉토리/파일 기능, 흐름도 및 아키텍처 기술.

---

## 주요 코드 흐름 (파이프라인)

1. **main.py**에서 config.yaml 로드 → 데이터 준비(data/loader.py, preprocessing.py, dataset.py) → 모델 초기화(model/graph_network.py, diffusion.py) → 학습(train.py) 또는 샘플 생성(generate.py) → 결과 저장 및 시각화(utils/visualization.py).
2. 각 단계별 함수/클래스 명칭은 일관성 있게 관리. config key와 코드 변수명, 반환 값 타입, 인자 전달 등 명확하게 구분.
3. 데이터 흐름: loader → preprocessing → dataset → train/generate로 타입과 반환 값 체크 필수.
4. 모델/디퓨전 연결: diffusion.py와 graph_network.py 함수 인자, 반환값 타입 일치 확인.
5. 학습/생성 결과 저장 및 시각화 시 경로/파일명, 변수명 오류 방지.

---

## AI 프롬프트 및 자동화에 최적화된 설명

- 각 디렉토리/파일의 책임과 역할을 명확히 분리.
- 함수/클래스 명은 glossary 및 AI_CODING_GUIDELINES.md에 정의된 규칙을 따름.
- 데이터 → 모델 → 학습/생성 → 시각화 전 과정이 명확하게 파이프라인으로 연결됨.
- 프롬프트 작성 시 각 단계, 함수, 변수명을 일관성 있게 기술해야 AI가 정확히 이해하고 자동화/생성 가능.

# 체크리스트
config에서 변수명 일관성: config.yaml의 key와 코드 내 변수명이 일치하도록 주의.
데이터 흐름: loader → preprocessing → dataset → train/generate로 데이터 타입이 바뀔 수 있으니, return 타입 체크!
model/diffusion 연결: diffusion.py와 graph_network.py의 함수 인자, 반환값 타입이 맞는지 확인.
train/generate 인자 전달: 모델, 데이터셋, config 등 필요한 인자를 빠짐없이 전달.
utils/config 파싱: config 불러올 때 경로/파일명 실수 주의.
시각화 함수 호출: 결과 저장 경로, 파일명 변수 실수 없도록.
