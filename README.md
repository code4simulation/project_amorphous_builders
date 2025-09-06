# project_amorphous_builders

# Prompt
비정질 원자 구조를 diffusion model을 통해서 생성하고 싶어. 순차적으로 프로그래밍을 해보려고 해.

학습 데이터 준비: extxyz 포맷으로 원자 구조에 대한 학습 데이터 제공, graph neural network 표현 방식으로 데이터 큐레이션
학습: 데이터를 기반으로 diffusion model 학습
생성: lattice vector, density, chemical formula, number of atoms를 기본적인 구조 셋업 인자로 받기, target property(ex. parital RDFs) 인자로 받아서 conditional diffusion model로 원자 구조 생성.
인풋 셋업: yaml 형태로 학습 준비, 학습 또는 생성에 대한 인자를 "input.yaml"에서 제어

1) 위 설명에 맞춰 인풋 파일과 파이썬 코드 아키텍처의 모식도를 생성해줘
2) "init과 main를 비롯하여 '학습 데이터 준비:extxyz 포맷으로 원자 구조에 대한 학습 데이터 제공, graph neural network 표현 방식으로 데이터 큐레이션' 단계에 해당하는 'src/data' 내부에 loader.py, preprocessing.py, dataset.py를 작성해줘.
이 때 코드 내부에서 중간 결과 체크를 위해 로그를 남기는 것은 print 말고 로그 파일을 작성하는 방향으로 코드를 작성해."
3) src/data 코드를 기반으로 extxyz 파일을 입력해서 데이터가 처리되는 과정이 정상적으로 작동하는지 확인하고 싶어. 어떻게 코드를 실행하면 될지 알려줘.
4) asdf
'학습: target property와 preprocessed 데이터를 기반으로 conditional diffusion model 학습' 단계에 해당하는 train.py와 model 디렉토리 내부의 파이썬 코드를 작성해줘.
이 때 코드 내부에서 중간 결과 체크를 위해 로그를 남기는 것은 print 말고 로그 파일을 작성하는 방향으로 코드를 작성해.
5)
(단계1)학습 데이터 준비: extxyz 포맷으로 원자 구조에 대한 학습 데이터 제공, graph neural network 표현 방식으로 데이터 큐레이션
(단계2)학습: 데이터를 기반으로 conditional diffusion model 학습
위 두 단계에 대해 input.yaml을 파싱하는 utils/config.py를 작성해줘.
src/main.py 를 통해 input.yaml의 파라미터를 기준으로 프로그램을 실행하는 커맨드 예시까지 알려줘.

# Schematic Code Archtecture
```
project_amorphous_builders/
│
├── examples/
│   └── ...          # 실행 예시, 샘플 config, 사용 방법 등 실험/테스트 케이스를 둔다.
│
├── src/
│   ├── __init__.py  # src 패키지 초기화
│   ├── main.py      # 주요 실행 진입점. config, 데이터 로드, 모델 초기화/학습/생성 전체 파이프라인 제어.
│
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # extxyz 등 원자 구조 데이터 파일 로드. 데이터 경로, 포맷 파싱, 데이터프레임/배열 반환.
│   │   ├── preprocessing.py  # 데이터 전처리: 정규화, 증강, 샘플링, 마스킹 등 모델 입력 전 가공.
│   │   └── dataset.py        # PyTorch Dataset/Loader 클래스 정의. 학습/테스트 데이터 관리, 배치 생성.
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── graph_network.py  # GNN 모델 정의. U-Net 구조, Graph Attention 레이어 등 핵심 신경망 구성.
│   │   └── diffusion.py      # Diffusion process 코드. Forward/reverse step 함수, noise 추가/제거, 샘플 생성.
│   │
│   ├── train.py      # 학습 루프/로직. 모델, 데이터셋, 옵티마이저 초기화, 에폭/스텝 반복, 체크포인트 저장/로깅.
│   ├── generate.py   # 학습된 모델로 새로운 원자 구조 생성. diffusion sampling, 결과 저장, 시각화 등.
│
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py          # YAML 등 설정 파일 파싱. 하이퍼파라메터, 경로, 옵션 관리.
│   │   └── visualization.py   # 결과 시각화: 원자 구조 plot, 학습 곡선, 샘플 이미지 등.
│
└── ...
```
# 파일별 흐름과 함수/변수 연결 코멘트
1. main.py
전체 파이프라인 제어.
config.yaml 로드 → 데이터셋 준비(data/loader.py, preprocessing.py, dataset.py) → 모델 초기화(model/graph_network.py) → 학습(train.py) 또는 샘플 생성(generate.py).
함수/변수 연결에서, config 파라미터를 각 단계로 올바르게 전달하는 것이 중요.
예: main(config_path) → config = utils.config.load_config(config_path) → dataset = data.loader.load_data(config.data) 등.
2. data/loader.py
원자 구조 파일(extxyz 등)을 로드하여 numpy/pandas/torch 형식으로 변환.
로드된 데이터를 preprocessing.py로 넘김.
변수명 예: load_extxyz(path) → 반환값: 구조 배열/좌표/원자종 등.
3. data/preprocessing.py
정규화, 증강, 마스킹 등 데이터 가공.
loader.py에서 받은 raw 데이터를 가공하여 dataset.py로 전달.
함수 예: normalize(data), augment(data)
4. data/dataset.py
PyTorch Dataset 클래스를 구현.
preprocessing을 거친 데이터를 배치 단위로 반환.
학습/생성 단계에서 반복자(iterator)로 사용.
5. model/graph_network.py
GNN, U-Net, Graph Attention 등 네트워크 정의.
인자: 원자 구조, 그래프 정보, timestep, noise 등.
변수/함수 이름 일관성 필요 (예: forward(x, edge_index, t)).
6. model/diffusion.py
Diffusion process: Forward/Reverse step 구현.
noise 추가, 제거, 샘플 생성.
모델의 forward와 연결됨.
주요 함수: forward_step(x, t), reverse_step(x, t)
7. train.py
학습 루프 제어.
데이터셋, 모델, 옵티마이저, 로스함수 등 초기화.
에폭/스텝 반복, 체크포인트 저장.
인자 전달: 데이터셋, 모델, config 등.
8. generate.py
학습된 모델 불러와서 새로운 구조 생성.
diffusion sampling, 결과 저장 및 시각화(utils/visualization.py 호출).
인자: 모델, 샘플링 파라미터, 저장 경로 등.
9. utils/config.py
yaml/json 등 설정 파일 파싱.
전체 파이프라인의 하이퍼파라미터, 경로, 옵션 관리.
함수 예: load_config(path) → dict 반환.
10. utils/visualization.py
원자 구조 plot, 학습 곡선, 생성 결과 등 시각화 함수.
matplotlib, plotly 등 사용 가능.

# 체크리스트
config에서 변수명 일관성: config.yaml의 key와 코드 내 변수명이 일치하도록 주의.
데이터 흐름: loader → preprocessing → dataset → train/generate로 데이터 타입이 바뀔 수 있으니, return 타입 체크!
model/diffusion 연결: diffusion.py와 graph_network.py의 함수 인자, 반환값 타입이 맞는지 확인.
train/generate 인자 전달: 모델, 데이터셋, config 등 필요한 인자를 빠짐없이 전달.
utils/config 파싱: config 불러올 때 경로/파일명 실수 주의.
시각화 함수 호출: 결과 저장 경로, 파일명 변수 실수 없도록.
