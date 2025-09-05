from ase import Atoms
from ase.io import write
import numpy as np

def create_test_structures():
    structures = []
    
    # example - SiO2
    for i in range(5):
        num_atoms = 90 

        positions = np.random.rand(num_atoms, 3) * 10.0
        symbols = ['Si'] * (num_atoms // 3) + ['O'] * (2 * num_atoms // 3)
        
        atoms = Atoms(symbols=symbols, 
                     positions=positions,
                     cell=[10.0, 10.0, 10.0],
                     pbc=True)
        
        structures.append(atoms)
    
    return structures

structures = create_test_structures()
write('test_structures.extxyz', structures, format='extxyz')
print("테스트 데이터 생성 완료: test_structures.extxyz")
