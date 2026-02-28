import torch, os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vanilla_gan_model import build_gan
from data_loader_signatures import SignatureDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 100
EPOCHS = 200
BATCH_SIZE = 64
LR = 2e-4

dataset = SignatureDataset("data/cedar")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G, D = build_gan(Z_DIM)
G, D = G.to(DEVICE), D.to(DEVICE)

optG = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), LR, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

os.makedirs("samples/generic", exist_ok=True)
os.makedirs("checkpoints/generic", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    for real in loader:
        real = real.to(DEVICE)
        bs = real.size(0)

        z = torch.randn(bs, Z_DIM, device=DEVICE)
        fake = G(z)

        real_lbl = torch.ones(bs, 1, device=DEVICE) * 0.9
        fake_lbl = torch.zeros(bs, 1, device=DEVICE)

        optD.zero_grad()
        d_loss = criterion(D(real), real_lbl) + criterion(D(fake.detach()), fake_lbl)
        d_loss.backward()
        optD.step()

        optG.zero_grad()
        g_loss = criterion(D(fake), real_lbl)
        g_loss.backward()
        optG.step()

    if epoch % 10 == 0:
        save_image(fake[:16], f"samples/generic/epoch_{epoch}.png", normalize=True)
        print(f"[Generic] Epoch {epoch}/{EPOCHS} | D:{d_loss.item():.3f} G:{g_loss.item():.3f}")

torch.save(G.state_dict(), "checkpoints/generic/generator.pth")
