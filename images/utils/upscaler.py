from pathlib import Path
from PIL import Image
import numpy as np
import torch
import time

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
UPSCALERS_DIR = BASE_DIR / "upscalers"

# Modelos disponibles
UPSCALE_MODELS = {
    "realistic": {
        "name": "RealESRGAN_x4plus",
        "file": "RealESRGAN_x4plus.pth",
        "scale": 1.75
    },
    "anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 1.75
    },
    "general": {
        "name": "realesr-general-x4v3",
        "file": "realesr-general-x4v3.pth",
        "scale": 1.75
    },
}

def apply_upscale(image: Image.Image, mode: str = "realistic") -> Image.Image:
    if mode not in UPSCALE_MODELS:
        raise ValueError(f"Modo '{mode}' no válido. Usa: {list(UPSCALE_MODELS.keys())}")

    config = UPSCALE_MODELS[mode]
    model_path = UPSCALERS_DIR / config["file"]
    internal_scale = config["scale"]
    desired_scale = 1.99

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo de upscale: {model_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Selección de arquitectura según nombre del modelo
    if config["name"] == "realesr-general-x4v3":
        print("📐 Arquitectura detectada: SRVGGNetCompact")
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type='prelu'
        )
    elif config["name"] == "RealESRGAN_x4plus_anime_6B":
        print("📐 Arquitectura detectada: RRDBNet (anime, 6 bloques)")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=6,
            num_grow_ch=32
        )
    elif config["name"] == "RealESRGAN_x4plus":
        print("📐 Arquitectura detectada: RRDBNet (realistic, 23 bloques)")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32
        )

    # Cargar pesos si hay modelo explícito
    if model is not None:
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict):
                if "params_ema" in state_dict:
                    state_dict = state_dict["params_ema"]
                elif "params" in state_dict:
                    state_dict = state_dict["params"]
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Modelo '{mode}' cargado con strict=False en {device}")
        except Exception as e:
            raise RuntimeError(f"❌ Error al cargar pesos para '{mode}': {e}")
    else:
        print(f"📦 Modelo '{mode}' se cargará automáticamente en RealESRGANer")

    # Inicializar el upsampler
    upsampler = RealESRGANer(
        model_path=str(model_path),
        model=model,
        scale=internal_scale,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )

    # Aplicar upscale
    img_np = np.array(image.convert("RGB"))
    print(f"🔧 Iniciando upscale con outscale={desired_scale}...")
    start_time = time.time()

    try:
        output_np, _ = upsampler.enhance(img_np, outscale=desired_scale)
    except Exception as e:
        raise RuntimeError(f"❌ Error durante upscale con RealESRGANer: {e}")

    end_time = time.time()
    print(f"✅ Upscale finalizado en {end_time - start_time:.2f} s")

    upscaled = Image.fromarray(output_np)

    # Reescalar si es necesario
    if internal_scale > desired_scale:
        new_size = tuple((s * desired_scale // internal_scale) for s in upscaled.size)
        print(f"🔧 Reescalando de {upscaled.size} a {new_size} (forzado x2 final)")
        upscaled = upscaled.resize(new_size, Image.LANCZOS)

    return upscaled
