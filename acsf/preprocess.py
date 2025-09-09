import numpy as np 
import os
from ase.io import read
from dscribe.descriptors import ACSF
from collections import defaultdict

def preprocess_amorphous_structures(data_folder, species, rcut, rdf_rmax, rdf_bin_width):
    """
    extxyz 파일들로부터 ACSF 특징 벡터와 평균 Partial RDF를 계산합니다.

    Args:
        data_folder (str): .extxyz 파일들이 있는 폴더 경로
        species (list): 데이터에 포함된 모든 원소의 원자 번호 리스트 (예: [14, 8] for SiO2)
        rcut (float): ACSF 계산을 위한 cutoff 반경
        rdf_rmax (float): RDF 계산을 위한 최대 거리
        rdf_bin_width (float): RDF 계산을 위한 거리 bin의 너비

    Returns:
        None. 파일로 결과를 저장합니다.
    """
    print("데이터 전처리를 시작합니다...")

    # 1. ACSF Descriptor 설정
    acsf = ACSF(
        species=species,
        r_cut=rcut,
        g2_params=[[1, 1], [1, 2], [1, 3]], # 예시 파라미터, 데이터에 맞게 조정 필요
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]], # 예시 파라미터
        periodic=False # 비정질 구조이므로 주기성 비활성화
    )

    all_acsf_features = []
    
    # Partial RDF 계산을 위한 변수
    rdf_bins = np.arange(0, rdf_rmax + rdf_bin_width, rdf_bin_width)
    rdf_r_values = (rdf_bins[:-1] + rdf_bins[1:]) / 2
    
    # {('Si', 'O'): [카운트], ...} 형태
    total_rdf_histograms = defaultdict(lambda: np.zeros(len(rdf_bins) - 1))
    pair_counts = defaultdict(int)
    total_atoms = defaultdict(int)
    num_structures = 0

    extxyz_files = [f for f in os.listdir(data_folder) if f.endswith('.extxyz')]
    if not extxyz_files:
        print(f"경고: '{data_folder}' 폴더에 .extxyz 파일이 없습니다.")
        return

    # 2. 각 파일을 순회하며 데이터 처리
    for i, filename in enumerate(extxyz_files):
        filepath = os.path.join(data_folder, filename)
        atoms = read(filepath, format='extxyz')
        num_structures += 1
        
        # --- ACSF 특징 벡터 생성 ---
        # 각 원자에 대한 특징 벡터를 생성 (n_atoms, n_features)
        acsf_features = acsf.create(atoms)#, positions=atoms.get_positions())
        all_acsf_features.append(acsf_features)
        
        # --- Partial RDF 계산을 위한 데이터 누적 ---
        symbols = atoms.get_chemical_symbols()
        for sym in species:
             total_atoms[sym] += symbols.count(str(sym))

        distances = atoms.get_all_distances(mic=True) # mic=True는 최소 이미지 규약 적용
        
        for i_atom in range(len(atoms)):
            for j_atom in range(i_atom + 1, len(atoms)):
                dist = distances[i_atom, j_atom]
                
                # 거리(r)가 RDF 계산 범위 내에 있을 경우
                if dist < rdf_rmax:
                    # 원자 쌍 정의 (알파벳 순서로 정렬하여 일관성 유지)
                    pair = tuple(sorted((symbols[i_atom], symbols[j_atom])))
                    
                    # 해당 거리(r)가 속하는 bin의 인덱스를 찾아 히스토그램 업데이트
                    bin_index = int(dist / rdf_bin_width)
                    total_rdf_histograms[pair][bin_index] += 2 # 쌍이므로 2를 더함
                    pair_counts[pair] += 2
        
        print(f" - 파일 처리 완료: {filename} ({i+1}/{len(extxyz_files)})")


    # 3. 데이터셋 전체의 평균 Partial RDF 계산 및 정규화
    final_partial_rdfs = {}
    box_volume = np.mean([read(os.path.join(data_folder, f)).get_volume() for f in extxyz_files]) # 평균 부피
    
    for pair, histogram in total_rdf_histograms.items():
        symbol1, symbol2 = pair
        # g(r) = (V / N1*N2) * (dN / 4*pi*r^2*dr)
        # N_ideal = 4 * pi * r^2 * dr * (N1*N2 / V)
        n1 = total_atoms[symbol1]
        n2 = total_atoms[symbol2]
        
        if symbol1 == symbol2:
            num_pairs = n1 * (n1 - 1) / 2
        else:
            num_pairs = n1 * n2

        number_density = num_pairs / box_volume / num_structures
        
        # 각 bin의 부피 (4 * pi * r^2 * dr)
        shell_volumes = 4 * np.pi * rdf_r_values**2 * rdf_bin_width
        
        # 이상적인 경우의 원자 수
        ideal_counts = shell_volumes * number_density

        # 0으로 나누는 것을 방지
        ideal_counts[ideal_counts == 0] = 1e-9

        g_r = histogram / ideal_counts / num_structures
        
        # {('Si', 'O'): [r 배열, g(r) 배열]} 형태로 저장
        final_partial_rdfs[pair] = [rdf_r_values, g_r]


    # 4. 결과 파일 저장
    print("\n계산된 특징 벡터와 RDF를 파일로 저장합니다...")
    # ACSF 특징 벡터 저장 (리스트의 리스트 형태)
    np.save('acsf_features.npy', np.array(all_acsf_features, dtype=object), allow_pickle=True)
    # Partial RDF 저장 (딕셔너리 형태)
    np.save('partial_rdfs.npy', final_partial_rdfs)

    print(f"\n전처리 완료!")
    print(f" - ACSF 특징 벡터 저장: acsf_features.npy")
    print(f" - Partial RDF 저장: partial_rdfs.npy")
    print(f" - 처리된 구조 개수: {num_structures}")
    print(f" - 생성된 Partial RDF 쌍: {list(final_partial_rdfs.keys())}")


# --- 스크립트 실행 ---
if __name__ == '__main__':
    # 사용자 설정 변수
    DATA_FOLDER = './' # extxyz 파일이 있는 폴더 경로
    
    # 예시: SiO2의 경우, Si=14, O=8
    # 본인의 데이터에 맞게 원자 번호 목록을 수정하세요.
    ELEMENTS = ['Si']
    
    ACSF_RCUT = 6.0 # ACSF 계산 시 Cutoff (Angstrom)
    RDF_R_MAX = 10.0 # RDF 계산 최대 거리 (Angstrom)
    RDF_BIN_WIDTH = 0.05 # RDF 거리 bin 너비 (Angstrom)

    # 함수 실행
    preprocess_amorphous_structures(
        data_folder=DATA_FOLDER,
        species=ELEMENTS,
        rcut=ACSF_RCUT,
        rdf_rmax=RDF_R_MAX,
        rdf_bin_width=RDF_BIN_WIDTH
    )
