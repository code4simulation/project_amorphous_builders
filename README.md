# project_amorphous_builders

# Schematic Code Archtecture
examples/
src/
├── __init__.py           # Initialization
├── main.py               # main
├── data/
│ ├── __init__.py
│ ├── loader.py           # extxyz data load
│ ├── preprocessing.py    #
│ └── dataset.py          # 
├── model/
│ ├── __init__.py
│ ├── graph_network.py    # GNN model (U-Net + Graph Attention)
│ └── diffusion.py        # Diffusion (Forward/Reverse)
├── train.py              # 
├── generate.py           # 
└── utils/
    ├── __init__.py
    ├── config.py         # parsing YAML
    └── visualization.py  # 
