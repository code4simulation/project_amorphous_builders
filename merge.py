import numpy as np
from ase import Atoms
import random
from typing import Dict, Tuple

# 이 함수는 이미 파일에 존재한다고 가정합니다.
# from .differentiable_rdf import differentiable_partial_rdf

def _calculate_cost(current_prdfs: Dict[str, np.ndarray], target_prdfs: Dict[str, np.ndarray]) -> float:
    """
    현재 pRDF와 목표 pRDF 간의 평균 제곱 오차(MSE)를 계산합니다.
    """
    total_error = 0.0
    num_pairs = 0
    for pair_key, target_rdf in target_prdfs.items():
        if pair_key in current_prdfs:
            current_rdf = current_prdfs[pair_key]
            # r 값들이 동일하다고 가정하고 g(r) 값 (1번 열)을 비교합니다.
            if current_rdf.shape == target_rdf.shape:
                # g(r) 값의 차이에 대한 제곱합을 계산합니다.
                error = np.sum((current_rdf[:, 1] - target_rdf[:, 1])**2)
                total_error += error
                num_pairs += 1
    
    return total_error / num_pairs if num_pairs > 0 else 0.0

def reverse_monte_carlo_prdf(
    atoms: Atoms,
    prdfs: Dict[str, np.ndarray],
    n_steps: int,
    dr_max: float,
    temperature: float,
    r_max: float,
    n_bins: int,
    verbose: bool = False
) -> Atoms:
    """
    Reverse Monte Carlo (RMC) 방법을 사용하여 목표 부분 RDF(prdf)에 맞도록 원자 구조를 최적화합니다.

    Args:
        atoms (ase.Atoms): 최적화를 시작할 초기 원자 구조.
        prdfs (Dict[str, np.ndarray]): 목표 pRDF 데이터 딕셔너리.
            - Key: "Si-Si", "Si-O"와 같은 원소 쌍 문자열.
            - Value: (n_bins, 2) 형태의 NumPy 배열 [r, g(r)].
        n_steps (int): 수행할 총 몬테카를로 스텝 수.
        dr_max (float): 단일 MC 이동 시 원자의 최대 변위.
        temperature (float): Metropolis 기준에 사용될 유효 온도 (kT).
        r_max (float): RDF 계산 시 최대 반경.
        n_bins (int): RDF 계산 시 사용할 빈(bin)의 수.
        verbose (bool): True일 경우, 시뮬레이션 진행 상황을 출력합니다.

    Returns:
        ase.Atoms: 최적화된 원자 구조.
    """
    current_atoms = atoms.copy()
    current_atoms.set_pbc(True)
    num_atoms = len(current_atoms)

    # 초기 pRDF와 비용 계산
    element_pairs = [tuple(key.split('-')) for key in prdfs.keys()]
    
    current_prdfs = {}
    for pair in element_pairs:
        key = f"{pair[0]}-{pair[1]}"
        # 제공된 differentiable_partial_rdf 함수를 사용하여 RDF 계산
        rdf = differentiable_partial_rdf(current_atoms, r_max=r_max, n_bins=n_bins, elements=pair)
        current_prdfs[key] = rdf
    
    current_cost = _calculate_cost(current_prdfs, prdfs)
    
    if verbose:
        print(f"초기 비용: {current_cost:.6f}")

    # RMC 루프
    for step in range(n_steps):
        # 1. 임의의 원자 선택
        atom_idx = random.randint(0, num_atoms - 1)
        
        # 이전 위치 저장
        old_position = current_atoms.positions[atom_idx].copy()
        
        # 2. 임의의 변위 제안
        displacement = (np.random.rand(3) - 0.5) * 2 * dr_max
        current_atoms.positions[atom_idx] += displacement
        current_atoms.wrap()  # 원자를 주기 경계 조건에 맞게 셀 내부로 이동

        # 3. 새로운 pRDF와 비용 계산
        new_prdfs = {}
        for pair in element_pairs:
            key = f"{pair[0]}-{pair[1]}"
            rdf = differentiable_partial_rdf(current_atoms, r_max=r_max, n_bins=n_bins, elements=pair)
            new_prdfs[key] = rdf
        
        new_cost = _calculate_cost(new_prdfs, prdfs)
        
        # 4. Metropolis-Hastings 기준 적용
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        elif temperature > 0:
            # 비용이 증가한 경우, 확률적으로 수용
            prob = np.exp(-delta_cost / temperature)
            if random.random() < prob:
                accept = True
        
        # 5. 상태 업데이트
        if accept:
            current_cost = new_cost
        else:
            # 이동 거부: 원자 위치를 원래대로 복원
            current_atoms.positions[atom_idx] = old_position
            
        if verbose and (step + 1) % 100 == 0:
            print(f"스텝 [{step+1}/{n_steps}], 비용: {current_cost:.6f}, 수용 여부: {accept}")

    if verbose:
        print(f"최종 비용: {current_cost:.6f}")
        
    return current_atoms

