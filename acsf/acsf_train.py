import os 
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

class ACSFDataset(Dataset):
    def __init__(self, acsf_features_path):
        # (structures, atoms, features) numpy 배열 로드
        self.data = np.load(acsf_features_path, allow_pickle=True)
        # numpy object 배열일 경우 stacking 필요
        if isinstance(self.data[0], np.ndarray):
            self.data = np.array([d for d in self.data])
        # Tensor 변환
        self.data_tensor = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data_tensor) # 구조 수

    def __getitem__(self, idx):
        # (atoms, features) tensor를 반환
        return self.data_tensor[idx]

# 간단한 conditional diffusion model 예시 (입력: ACSF feature tensor)
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim), # input 크기와 동일
        )

    def forward(self, x):
        return self.net(x)

def train_conditional_diffusion(
    acsf_features_path,
    model_save_path="best_cond_diffusion.pth",
    epochs=100,
    batch_size=16,
    val_ratio=0.1,
    lr=1e-3,
    patience=10,
    save_interval=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = ACSFDataset(acsf_features_path)

    # train/val split
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # DataLoader: batch_size는 구조 단위(batch = 구조 개수)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ConditionalDiffusionModel(feature_dim=dataset.data_tensor.shape[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device) # (batch, atoms, features)
            # 예시: diffusion condition 별도 입력 없이 reconstruction loss 만 계산
            pred = model(batch)
            loss = criterion(pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss_avg = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch)
                val_losses.append(loss.item())
        val_loss_avg = sum(val_losses) / len(val_losses)

        # 로그 출력
        print(f"Epoch {epoch:03d} - Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss_avg:.6f}")

        # early stopping, best model 저장
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f" Best model saved with Val Loss: {val_loss_avg:.6f}")
        else:
            epochs_no_improve += 1

        # Early stopping 체크
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

        # 중간 상태 저장
        if epoch % save_interval == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Model checkpoint saved: {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    # 사용자 환경에 맞게 파일 경로 지정
    acsf_features_path = "acsf_features.npy"

    train_conditional_diffusion(
        acsf_features_path=acsf_features_path,
        epochs=100,
        batch_size=8, # 구조 수 단위 배치 크기 예시
        val_ratio=0.1,
        lr=1e-3,
        patience=10,
        save_interval=10,
    )
