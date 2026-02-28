import os
import io
import zipfile
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.generator_vanilla_gan import Generator

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 100
IMG_SIZE = 64

GENERIC_MODEL = "checkpoints/generic/generator.pth"
USER_SPECIFIC_DIR = "checkpoints/user_specific"

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Synthetic Signature Generator API",
    description="Generate synthetic handwritten signatures using Vanilla GAN",
    version="1.0"
)

# =========================
# REQUEST SCHEMA
# =========================
class GenerateRequest(BaseModel):
    n: int = 10
    user_id: str | None = None


# =========================
# UTILS
# =========================
def load_generator(user_id=None):
    G = Generator(Z_DIM).to(DEVICE)

    if user_id:
        model_path = os.path.join(USER_SPECIFIC_DIR, f"{user_id}.pth")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="User-specific model not found")
    else:
        model_path = GENERIC_MODEL
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Generic model not found")

    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()
    return G


def tensor_to_image(tensor):
    """
    Convert [-1,1] tensor → uint8 grayscale image
    """
    img = tensor.detach().cpu().squeeze().numpy()
    img = ((img + 1) / 2.0 * 255).astype(np.uint8)
    return img


# =========================
# API ENDPOINT
# =========================
@app.post("/generate")
def generate_signatures(req: GenerateRequest):

    if req.n <= 0 or req.n > 500:
        raise HTTPException(status_code=400, detail="n must be between 1 and 500")

    G = load_generator(req.user_id)

    noise = torch.randn(req.n, Z_DIM, device=DEVICE)

    with torch.no_grad():
        samples = G(noise)

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i in range(req.n):
            img = tensor_to_image(samples[i])

            img_bytes = io.BytesIO()
            from PIL import Image
            Image.fromarray(img).save(img_bytes, format="PNG")

            filename = f"sig_{i:03d}.png"
            zipf.writestr(filename, img_bytes.getvalue())

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=synthetic_signatures.zip"
        }
    )
