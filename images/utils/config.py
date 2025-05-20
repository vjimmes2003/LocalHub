from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
VAE_DIR = BASE_DIR / "vae"

NEGATIVE_PROMPT = (
    "nsfw, ns·fw, explicit, nudity, naked, nude, porn, hentai, ero, erotic, adult content, sex, sexual, sexualized, sensual, lewd, "
    "suggestive, revealing outfit, cleavage, see-through, underwear, lingerie, seduction, orgasm:1.3, "
    "nipple, areola, tits, breasts, busty, boobs, penis, dick, cock, vagina, pussy, ass, butt, anal, blowjob, cumshot, intercourse, bdsm, domination, submission:1.3, "
    "half-naked, topless, shirtless, exposed chest, panty, panties, bra, garterbelt, thighhighs, sex toys, dildo, exhibitionism:1.2, "
    "text, watermark, logo, signature, cropped, out of frame, jpeg artifacts, blurry, lowres, low quality, worst quality, "
    "deformed, mutated, bad anatomy, wrong anatomy, malformed limbs, fused fingers, too many fingers, missing limbs, disfigured, distorted, mutation, "
    "poorly drawn face, poorly drawn hands, disgusting, ugly, duplicate, morbid, mutilated, "
    "sketch, cartoon, drawing, anime, 3d, cgi, render:1.2"
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
        "default_resolution": (512, 512),
        "steps": 25,
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
        "steps": 35,
        "cfg_scale": 7.0,
        "sampler": "dpmpp_2m_karras"
    },
}
