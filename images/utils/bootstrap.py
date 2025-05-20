import os
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_BASE = BASE_DIR / "models"
UPSCALERS_BASE = BASE_DIR / "upscalers"
VAE_BASE = BASE_DIR / "vae"

RAW_MODEL_FILES = {
    "realisticvision-v6": {
        "url": "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1.safetensors",
        "dest": MODELS_BASE / "realisticvision-v6" / "Realistic_Vision_V6.0_NV_B1.safetensors"
    },
    "juggernautxl": {
        "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "dest": MODELS_BASE / "juggernautxl-v9" / "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    },
}

VAE = {
    "vae-ft-mse-840000-ema-pruned.safetensors":
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
}

UPSCALERS = {
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",

    "RealESRGAN_x4plus_anime_6B.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",

    "realesr-general-x4v3.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}


def download_file(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"✅ Ya existe: {dest_path.name}")
        return
    print(f"⬇️ Descargando: {dest_path.name}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Descargado: {dest_path.name}")

def check_models():
    for info in RAW_MODEL_FILES.values():
        download_file(info["url"], info["dest"])

def check_upscalers():
    for name, url in UPSCALERS.items():
        download_file(url, UPSCALERS_BASE / name)

def check_vae():
    for name, url in VAE.items():
        download_file(url, VAE_BASE / name)

def bootstrap_all():
    print("🔧 Iniciando verificación de modelos, upscalers y VAE...\n")
    check_models()
    check_vae()
    check_upscalers()
    print("\n✅ Todo listo.")

if __name__ == "__main__":
    bootstrap_all()
