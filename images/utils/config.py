from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
VAE_DIR = BASE_DIR / "vae"

NEGATIVE_PROMPT = (
    "nsfw, naked, nude, low quality, worst quality, jpeg artifacts, blurry, "
    "text, watermark, cropped, out of frame, "
    "bad anatomy, wrong anatomy, mutated, mutation, deformed, distorted, disfigured, "
    "poorly drawn face, poorly drawn hands, malformed limbs, extra limbs, missing limbs, fused fingers, too many fingers, long neck, "
    "duplicate, morbid, mutilated, ugly, disgusting, "
    "3d, cgi, sketch, cartoon, drawing, anime:1.2, nipple "
    "tits, breasts, penis, cock, half-naked, exhibitionism:0.8"
)


MODEL_CONFIGS = {
    "realisticvision-v6": {
        "name": "Realistic Vision V6.0 B1",
        "type": "sd15_safetensors",
        "path": MODELS_DIR / "realisticvision-v6" / "Realistic_Vision_V6.0_NV_B1.safetensors",
        "vae": {
            "path": VAE_DIR / "vae-ft-mse-840000-ema-pruned.safetensors"
        },
        "vram_required": "8 GB",
        "available_resolutions": [
            (512, 512),     # cuadrada básica
            (576, 768),     # vertical ligera
            (512, 896),     # vertical intermedia
            (640, 832),     # vertical media
            (768, 512),     # horizontal compacta
            (896, 512)      # horizontal intermedia
        ],
        "default_resolution": (768, 1024),
        "steps": 30,
        "cfg_scale": 7.0,
        "sampler": "dpmpp_2m_karras"
    },
    "juggernautxl": {
        "name": "Juggernaut XL v9",
        "type": "sdxl_safetensors",
        "path": MODELS_DIR / "juggernautxl-v9" / "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "vae": None,
        "vram_required": "10-12 GB",
        "available_resolutions": [
            (1024, 1024),
            (1216, 832),
            (832, 1216)
        ],
        "default_resolution": (1024, 1024),
        "steps": 25,
        "cfg_scale": 2.0,
        "sampler": "dpmpp_sde_karras"
    },
}
