from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
import torch
import os
from images.utils.config import MODEL_CONFIGS
import time

def load_model(model_key):
    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]
    vae_path = config.get("vae", None)
    model_type = config["type"]

    print(f"\n📦 Cargando modelo: {model_key}")
    print(f"🔍 Ruta del modelo: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ El archivo del modelo no existe: {model_path}")

    start = time.time()

    # Elegir la clase de pipeline según el tipo
    if model_type == "sdxl_safetensors":
        pipeline_class = StableDiffusionXLPipeline
    else:
        pipeline_class = StableDiffusionPipeline

    pipe = pipeline_class.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    )
    print(f"✅ Modelo {model_key} cargado en {round(time.time() - start, 2)}s")

    if vae_path:
        print(f"📦 Cargando VAE desde: {vae_path}")
        vae = AutoencoderKL.from_single_file(str(vae_path["path"]), torch_dtype=torch.float16)
        pipe.vae = vae
        print("✅ VAE asignado al pipeline.")
    else:
        print("⚠️ VAE no especificado, se usará el por defecto.")

    # Sampler
    sampler = config.get("sampler", "").lower()
    if sampler == "euler":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        print("⚙️ Sampler aplicado: Euler Ancestral")
    elif sampler == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        print("⚙️ Sampler aplicado: DPM++ 2M Karras")
    else:
        print("⚠️ Sampler no reconocido, se usa el default del modelo.")

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Modelo listo en dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    return pipe
