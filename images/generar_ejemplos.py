import os
from PIL import Image
import torch
import time

from images.utils.models import load_model
from images.utils.upscaler import apply_upscale
from images.utils.hires_fix import apply_hires_fix
from images.utils.config import MODEL_CONFIGS, NEGATIVE_PROMPT

# Crear carpeta de salida si no existe
os.makedirs("images/examples", exist_ok=True)

EXAMPLE_PROMPTS = {
    "realisticvision-v6": [
        "instagram photo, closeup face photo of 23 y.o Chloe in black sweater, cleavage, pale skin, (smile:0.4), hard shadows",
        "closeup face photo of caucasian man in black clothes, night city street, bokeh",
        "instagram photo, front shot, portrait photo of a 24 y.o woman, wearing dress, beautiful face, cinematic shot, dark shot",
        "oil painting, underwater image of an ancient city, glowing, by Greg Rutkowski",
        "polaroid photo of a road, warm tones, perfect landscape"
    ],
    "juggernautxl": [
        "beautiful lady, (freckles), big smile, ruby eyes, long curly hair, dark makeup, hyperdetailed photography, soft light, head and shoulders portrait, cover",
        "Leica portrait of a gremlin skateboarding, coded patterns, sparse and simple, uhd image, urbancore, sovietwave, period snapshot",
        "A hyperdetailed photograph of a Cat dressed as a mafia boss holding a fish walking down a Japanese fish market with an angry face, 8k resolution, best quality, beautiful photograph, dynamic lighting",
        "photograph, a path in the woods with leaves and the sun shining, by Julian Allen, dramatic autumn landscape, rich cold moody colours, hi resolution",
        "a torso with a TV instead of a head"
    ]
}

def generar_imagen(prompt, model_key, idx):
    print(f"\n--- Generando ejemplo {model_key} #{idx+1} ---")
    config = MODEL_CONFIGS[model_key]
    pipe = load_model(model_key)

    steps = config["steps"]
    cfg_scale = config["cfg_scale"]
    width, height = config["default_resolution"]

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    # Generar imagen base
    start = time.time()
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
        negative_prompt=NEGATIVE_PROMPT
    )
    base_image = result.images[0]
    print(f"🕒 Imagen base generada en {time.time() - start:.2f}s")

    # Guardar imagen base
    base_path = f"images/examples/{model_key}_{idx+1}_base.png"
    base_image.save(base_path)
    print(f"✅ Guardado: {base_path}")

    # Mejorar con Hires.Fix o Upscaler
    if config["type"] == "sd15_safetensors":
        print("✨ Aplicando Hires.Fix...")
        final_image = apply_hires_fix(
            image=base_image,
            prompt=prompt,
            model_key=model_key,
            seed=42,
            denoising_strength=0.25,
            upscale_factor=1.5
        )
    else:
        print("🔍 Aplicando Upscaler...")
        final_image = apply_upscale(base_image, mode="anime")

    # Guardar imagen mejorada
    final_path = f"images/examples/{model_key}_{idx+1}_final.png"
    final_image.save(final_path)
    print(f"✅ Guardado: {final_path}")

# Ejecutar todo
for model_key, prompts in EXAMPLE_PROMPTS.items():
    for idx, prompt in enumerate(prompts):
        generar_imagen(prompt, model_key, idx)
