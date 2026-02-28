import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from siamese_model import SiameseCNN
from signature_pairs_dataset import SignaturePairsDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(use_gan=False):
    dataset = SignaturePairsDataset(
        root_dir="data/cedar",
        use_gan=use_gan,
        gan_dir="generated/generic" if use_gan else None
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SiameseCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        total_loss = 0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE).unsqueeze(1)

            output = model(img1, img2)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    name = "augmented" if use_gan else "baseline"
    torch.save(model.state_dict(), f"checkpoints/siamese_{name}.pth")

if __name__ == "__main__":
    print("Training baseline verifier...")
    train(use_gan=False)

    print("\nTraining GAN-augmented verifier...")
    train(use_gan=True)
