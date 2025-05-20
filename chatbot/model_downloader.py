# chatbot/model_downloader.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_CONFIGS, CACHE_DIR

def ensure_models_downloaded():
    for name, config in MODEL_CONFIGS.items():
        repo_id = config["repo_id"]
        model_dir = os.path.join(CACHE_DIR, name)

        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"📥 Descargando modelo '{name}' desde {repo_id}...")

            # Permitir código remoto solo si es necesario
            trust_remote = "instella" in repo_id.lower() or "trust_remote_code" in config.get("pipeline_kwargs", {})

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                cache_dir=model_dir,
                trust_remote_code=trust_remote,
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                cache_dir=model_dir,
                trust_remote_code=trust_remote,
            )

            print(f"✅ Modelo '{name}' descargado y guardado en {model_dir}")
        else:
            print(f"✅ Modelo '{name}' ya existe en caché.")
