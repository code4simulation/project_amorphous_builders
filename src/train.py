def run_training(dataset, model, config, save_dir):
    """
    Main training loop for diffusion model.
    Args:
        dataset: AmorphousDataset instance (PyTorch Dataset)
        model: DiffusionProcess instance
        config: Dictionary of training parameters (epochs, batch_size, optimizer, etc.)
        save_dir: Directory to save checkpoints and logs
    """
    import torch
    from torch.utils.data import DataLoader
    import os

    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint
        if (epoch+1) % config.get('checkpoint_interval', 10) == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    print("Training completed.")
