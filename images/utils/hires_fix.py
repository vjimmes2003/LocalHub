import torch
from PIL import Image
import os
import cv2
import numpy as np

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline
)

from images.utils.config import MODEL_CONFIGS

def upscale_cv2(image: Image.Image, scale=2) -> Image.Image:
    img_np = np.array(image.convert("RGB"))  # asegúrate de estar en RGB
    h, w = img_np.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    print(f"🔍 Escalando imagen de ({w}, {h}) a {new_size} con OpenCV + INTER_CUBIC...")
    img_upscaled = cv2.resize(img_np, new_size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(img_upscaled)

def apply_hires_fix(
    prompt: str,
    model_key: str,
    image: Image.Image,
    seed: int = None,
    denoising_strength: float = 0.45,
    upscale_factor: float = 2.0
) -> Image.Image:
    """
    Aplica Hires.Fix adaptativamente según el modelo (SD 1.5 o SDXL).

    Retorna una imagen mejorada.
    """
    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]
    model_type = config["type"]
    guidance_scale = config["cfg_scale"]
    steps = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Escalar con OpenCV (en lugar de PIL)
    image_resized = upscale_cv2(image, upscale_factor)

    # Elegir el pipeline adecuado
    generator = torch.Generator(device)
    if seed is not None:
        generator.manual_seed(seed)

    if model_type == "sdxl_safetensors":
        print("⚙️ Usando Hires.Fix con SDXL (img2img pipeline)...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        ).to(device)

    elif model_type == "sd15_safetensors":
        print("⚙️ Usando Hires.Fix con SD 1.5...")
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        ).to(device)

    else:
        print(f"⚠️ Hires.Fix no compatible con el modelo '{model_type}'.")
        return image

    result = pipe(
        prompt=prompt,
        image=image_resized,
        strength=denoising_strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        negative_prompt=config.get("negative_prompt", None),
    )

    print(f"✅ Hires.Fix aplicado con éxito (resolución final: {result.images[0].size})")
    return result.images[0]