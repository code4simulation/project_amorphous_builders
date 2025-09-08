import argparse
import os
from src import train
from src.data import loader, preprocessing, dataset
from src.model import graph_network, diffusion
from src.utils import config, visualization

def main():
    parser = argparse.ArgumentParser(description="Project Amorphous Builders Workflow Controller")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Pipeline mode")
    args = parser.parse_args()

    # 1. Load config
    cfg = config.load_config(args.config)

    # 2. Data preparation
    raw_data = loader.load_extxyz(cfg['data']['extxyz_path'])
    processed_data = preprocessing.normalize(raw_data)
    ds = dataset.AmorphousDataset(processed_data)

    # 3. Model preparation
    if args.mode == "train":
        model = graph_network.ConditionalGraphNetwork(cfg['model'])
        diffusion_model = diffusion.DiffusionProcess(model, cfg['diffusion'])
        train.run_training(
            dataset=ds,
            model=diffusion_model,
            config=cfg['train'],
            save_dir=cfg['output']['save_dir']
        )
    elif args.mode == "generate":
        # Generate workflow (예시)
        model = graph_network.ConditionalGraphNetwork(cfg['model'])
        diffusion_model = diffusion.DiffusionProcess(model, cfg['diffusion'])
        # Load checkpoint
        diffusion_model.load_state(cfg['generate']['checkpoint_path'])
        samples = diffusion_model.generate(cfg['generate'])
        visualization.plot_samples(samples, cfg['output']['save_dir'])

if __name__ == "__main__":
    main()
