import os
import sys
import torch
import streamlit as st
from torchvision.utils import save_image
import numpy as np
# Fix imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.generator_vanilla_gan import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 100

st.set_page_config(page_title="Synthetic Signature Generator", layout="wide")
st.title("✍️ Synthetic Handwritten Signature Generator")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Generation Settings")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Generic", "User-Specific"]
)

user_id = None
if model_type == "User-Specific":
    user_id = st.sidebar.text_input("User ID (e.g., user_01)")

num_images = st.sidebar.slider("Number of Signatures", 1, 50, 10)

generate_btn = st.sidebar.button("Generate Signatures")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_generator(model_path):
    model = Generator(Z_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

if generate_btn:
    if model_type == "Generic":
        model_path = "checkpoints/generic/generator.pth"
    else:
        model_path = f"checkpoints/user_specific/{user_id}.pth"

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
    else:
        G = load_generator(model_path)

        z = torch.randn(num_images, Z_DIM, device=DEVICE)
        with torch.no_grad():
            samples = G(z)

        os.makedirs("samples/ui_output", exist_ok=True)
        save_image(samples, "samples/ui_output/generated.png", normalize=True)

        st.success("Signatures generated successfully!")

        cols = st.columns(5)
        for i in range(num_images):
            img = samples[i].detach().cpu().squeeze().numpy()
            img = ((img + 1) / 2.0 * 255).astype(np.uint8)
            cols[i % 5].image(
                img,
                caption=f"Sig {i+1}",
                clamp=True
            )

