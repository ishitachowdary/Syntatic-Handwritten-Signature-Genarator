import os
import torch
from torchvision.utils import save_image
from generator_vanilla_gan import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 100

def generate(model_path, out_dir, n=100):
    os.makedirs(out_dir, exist_ok=True)

    G = Generator(Z_DIM).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()

    with torch.no_grad():
        z = torch.randn(n, Z_DIM, device=DEVICE)
        samples = G(z)

    for i in range(n):
        save_image(samples[i], f"{out_dir}/sig_{i}.png", normalize=True)

if __name__ == "__main__":
    generate("checkpoints/generic/generator.pth", "generated/generic", 100)
