import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from siamese_model import SiameseCNN
from signature_pairs_dataset import SignaturePairsDataset
from utils.metrics import compute_far_frr_eer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path):
    dataset = SignaturePairsDataset("data/cedar")
    loader = DataLoader(dataset, batch_size=32)

    model = SiameseCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    scores, labels = [], []

    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            output = model(img1, img2)
            scores.extend(output.cpu().numpy().flatten())
            labels.extend(label.numpy())

    return compute_far_frr_eer(labels, scores)

if __name__ == "__main__":
    print("Baseline:")
    print(evaluate("checkpoints/siamese_baseline.pth"))

    print("\nWith GAN augmentation:")
    print(evaluate("checkpoints/siamese_augmented.pth"))
