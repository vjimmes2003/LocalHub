import torch
import os
import time
from PIL import Image
from images.utils.models import load_model
from images.utils.config import MODEL_CONFIGS, NEGATIVE_PROMPT
from images.utils.upscaler import apply_upscale
from images.utils.hires_fix import apply_hires_fix
from PIL import ImageFilter

# Asegura carpeta outputs
if not os.path.exists("images\\outputs"):
    os.makedirs("images\\outputs")

LOADED_MODELS = {}

def get_or_load_model(model_key: str):
    if model_key not in LOADED_MODELS:
        LOADED_MODELS[model_key] = load_model(model_key)
    return LOADED_MODELS[model_key]

def generate_image(prompt: str, model_key: str, resolution: tuple, seed: int = None, upscaler_key: str = "none"):

    start_total = time.perf_counter()

    timestamp = int(time.time())
    config = MODEL_CONFIGS[model_key]
    width, height = resolution
    steps = config["steps"]
    guidance_scale = config["cfg_scale"]
    pipe = get_or_load_model(model_key)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        generator.manual_seed(seed)
        print(f"🎲 Usando seed fija: {seed}")
    else:
        seed = generator.seed()
        print(f"🎲 Usando seed aleatoria: {seed}")

    print(f"\n🖼️ Generando imagen con:")
    print(f"📌 Modelo: {model_key}")
    print(f"📐 Resolución: {width}x{height}")
    print(f"⚙️ Steps: {steps} | CFG: {guidance_scale}")
    print(f"🧠 Upscaler: {upscaler_key}")

    # Generación
    start_pipe = time.perf_counter()
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        negative_prompt=NEGATIVE_PROMPT
    )
    print(f"🕒 Tiempo generación base: {time.perf_counter() - start_pipe:.2f} s")

    image_base = result.images[0]
    filename_base = f"output_{model_key}_{timestamp}_base.png"
    save_path_base = os.path.join("images\\outputs", filename_base)
    image_base.save(save_path_base, format="PNG")
    
    # Aplicar hires.fix solo a SD1.5
    if config["type"] == "sd15_safetensors":
        print("✨ Aplicando Hires.Fix...")
        try:
            start_hires = time.perf_counter()
            image_final = apply_hires_fix(
                image=image_base,
                prompt=prompt,
                model_key=model_key,
                seed=seed,
                denoising_strength=0.25,
                upscale_factor=1.5
            )
            print(f"✅ Hires.Fix aplicado con éxito")
            print(f"🕒 Tiempo Hires.Fix: {time.perf_counter() - start_hires:.2f} s")
        except Exception as e:
            print(f"⚠️ Error en Hires.Fix: {e}")
            image_final = image_base
    else:
        print("⏩ Saltando Hires.Fix para modelos SDXL o no compatibles.")
        image_final = image_base

    # Upscaler si se indica
    if upscaler_key and upscaler_key.lower() != "none":
        print(f"🔍 Aplicando upscaler '{upscaler_key}'...")
        try:
            start_upscale = time.perf_counter()
            image_final = apply_upscale(image_final, upscaler_key)
            print(f"🕒 Tiempo upscale: {time.perf_counter() - start_upscale:.2f} s")
        except Exception as e:
            print(f"⚠️ Error aplicando upscale: {e}")

    # Guardar imagen final
    filename_final = f"output_{model_key}_{timestamp}_final.png"
    save_path_final = os.path.join("images\\outputs", filename_final)
    image_final.save(save_path_final, format="PNG")
    print(f"💾 Imagen guardada automáticamente en: {save_path_final}")

    print(f"🕒 Tiempo TOTAL: {time.perf_counter() - start_total:.2f} s")

    return image_base, image_final
