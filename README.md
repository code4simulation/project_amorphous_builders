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
\
examples/\
src/\
├── __init__.py           # Initialization\
├── main.py               # main\
├── data/\
│ ├── __init__.py\
│ ├── loader.py           # extxyz data load\
│ ├── preprocessing.py    #\
│ └── dataset.py          #\
├── model/\
│ ├── __init__.py\
│ ├── graph_network.py    # GNN model (U-Net + Graph Attention)\
│ └── diffusion.py        # Diffusion (Forward/Reverse)\
├── train.py              # \
├── generate.py           # \
└── utils/\
    ├── __init__.py\
    ├── config.py         # parsing YAML\
    └── visualization.py  #\
