# 📐 AI용 코딩 가이드라인 (for Generative AI Coding)

본 문서는 프로젝트에서 생성형 AI(Copilot, ChatGPT 등)를 활용한 코드 생성/설계/리뷰 시 **일관성(consistency)** 유지를 위한 규칙과 용어집(glossary)을 제공합니다.  
협업, 유지보수, 커뮤니케이션에서 혼란을 줄이고, AI가 컨텍스트에 따라 다르게 제안하는 변수명/함수명/구조를 표준화합니다.

---

## 1. **Naming Convention (이름 규칙)**

- **변수명/함수명**
  - 언어: 영어, 카멜케이스(camelCase) 또는 파이썬 표준(스네이크케이스 snake_case)
  - 의미가 명확한 단어 사용: `pos`, `edge_index`, `rdf`, `graph_cutoff` 등
  - 약어는 glossary에 반드시 등록, 혼용 금지
- **클래스명**
  - 파스칼케이스(PascalCase): `AmorphousDataset`, `ConditionalGraphNetwork`
- **디렉토리/파일명**
  - 모두 소문자, 하이픈(-)과 언더스코어(_) 혼용 금지
  - 기능/역할 기반 명명: `data/loader.py`, `model/graph_network.py`, `utils/config.py` 등

---

## 2. **디렉토리/아키텍처 구조 규칙**

```
project_root/
│
├── src/
│   ├── main.py              # 실행 진입점, 파이프라인 제어
│   ├── data/                # 데이터 입출력, 전처리, 데이터셋 관리
│   ├── model/               # 신경망, 네트워크, 확산모델 등
│   ├── utils/               # 설정, 시각화 등 보조 기능
│   └── ...                  # 기타 확장 모듈
├── examples/                # 샘플 config, 실험 코드
├── test/                    # 테스트 케이스
└── README.md                # 문서
```
- 각 디렉토리의 역할과 책임은 명확히 분리한다.
- import 경로는 src 기준 상대경로 사용.

---

## 3. **함수/클래스/모듈 역할 규칙**

- 함수명은 동작을 명확히 표현: `load_extxyz`, `batch_structures_to_graphs`, `calculate_rdf`, `normalize_rdf`, `create_dataset_from_structures`
- 클래스명은 데이터/기능 단위를 명확히: `AmorphousDataset`, `DiffusionProcess`
- 주요 파이프라인 단계별로 함수/클래스 분리 (데이터 로드 → 전처리 → 그래프 변환 → 모델 학습/생성)

---

## 4. **Style Guideline**

- docstring, 주석에 한글/영어 혼용 가능 (역할 명확히)
- 모든 함수/클래스에 타입 힌트, 반환값 명시
- 예외 처리, 로깅 필수 (`logger.info`, `logger.error`, 등)

---

## 5. **Glossary (용어집): 프로젝트 내 개념/약어 표준**

| 용어/약어         | 설명                                         | 예시 코드 변수명/함수명           |
|-------------------|----------------------------------------------|-----------------------------------|
| extxyz            | ASE의 확장 XYZ 원자 구조 파일                 | `extxyz_path`, `load_extxyz`      |
| graph             | PyG의 그래프 데이터 객체                     | `graph`, `batch_structures_to_graphs` |
| edge_index        | 그래프의 엣지 인덱스 ([2, E] shape)           | `edge_index`                      |
| edge_attr         | 엣지의 속성(거리 등) ([E, 1] shape)           | `edge_attr`                       |
| pos               | 원자 위치 ([N, 3] shape)                     | `pos`                             |
| atomic_numbers    | 원자 종류 (int array)                        | `atomic_numbers`                  |
| rdf               | Radial Distribution Function, RDF 특징 벡터   | `rdf`, `calculate_rdf`, `normalize_rdf` |
| graph_cutoff      | 그래프 생성 거리 임계값(Å)                    | `graph_cutoff`                    |
| n_bins            | RDF 히스토그램 bin 개수                       | `n_bins`                          |
| r_max             | RDF 최대 거리(Å)                              | `r_max`                           |
| Data              | PyG의 그래프 데이터 객체                      | `Data`, `AmorphousDataset`        |
| Dataset           | PyTorch/PyG 데이터셋 클래스                   | `AmorphousDataset`                |
| DiffusionProcess  | 확산모델의 forward/reverse step 관리 클래스   | `DiffusionProcess`                |
| ConditionalGraphNetwork | RDF 등 조건부 입력을 받는 그래프 신경망 | `ConditionalGraphNetwork`         |

---

## 6. **Prompt에 명시해야 하는 규칙 (AI 입력시)**
- 위 용어집/규칙에 따라 변수명/함수명을 일관적으로 사용하라고 명시
- 아키텍처/디렉토리 구조/클래스명 등은 반드시 기존 프로젝트 스타일을 따르라고 명시
- 새로 등장하는 개념/약어는 glossary에 추가할 것

---

## 7. **AI 코드 리뷰 체크리스트**

- [ ] 변수명/함수명/클래스명이 용어집과 일치하는가?
- [ ] 디렉토리/파일 구조가 프로젝트 기준과 일치하는가?
- [ ] 함수/클래스 역할이 명확히 분리되어 있는가?
- [ ] 타입, 예외, 로깅, docstring이 충분한가?
- [ ] 새로운 개념/약어가 등장하면 glossary에 등록되었는가?

---
