# chatbot/model.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path
import os
import gc

from .config import MODEL_CONFIGS, CACHE_DIR

# Nombres amigables usados en la interfaz => nombres reales usados en MODEL_CONFIGS y HuggingFace
MODEL_PATHS = {
    "Llama-3.2": "Llama-3.2-3B-Instruct",
    "Instella-3B": "Instella-3B-Instruct",
    "Qwen2.5": "Qwen2.5-3B-Instruct",
    "Stable-Code": "Stable-Code-Instruct-3B"
}

LOCAL_PATHS = {
    k: f"chatbot/models/{v}" for k, v in MODEL_PATHS.items()
}

_loaded_models = {}
_current_model = None 

def get_gpu_total_memory():
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory / (1024**2))  # en MiB
    return 0

def unload_model():
    global _loaded_models, _current_model
    if _current_model:
        print(f"🔻 Descargando modelo anterior: {_current_model}")
        del _loaded_models[_current_model]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _current_model = None

def load_model(name):
    global _loaded_models, _current_model
    
    if name in _loaded_models:
        return _loaded_models[name]
    
    unload_model()
    
    total_mem = get_gpu_total_memory()
    max_memory = {0: f"{int(total_mem * 0.9)}MiB"}
    internal_name = MODEL_PATHS.get(name, name)  # Traducción del nombre visible al interno
    config = MODEL_CONFIGS[internal_name]
    repo_id = config["repo_id"]
    task = config["task"]
    kwargs = config["pipeline_kwargs"]
    model_dir = f"{CACHE_DIR}/{internal_name}"

    print(f"📥 Cargando modelo '{internal_name}' desde {repo_id}...")

    # Algunos modelos requieren remote_code=True (como Instella)
    trust_remote = "Instella" in internal_name or config.get("trust_remote_code", False)
        
    tokenizer = AutoTokenizer.from_pretrained(repo_id,cache_dir=model_dir,trust_remote_code=trust_remote)
    model = AutoModelForCausalLM.from_pretrained(repo_id,cache_dir=model_dir,trust_remote_code=trust_remote,
        device_map="auto",            # reparte capas en GPU/CPU
        torch_dtype=torch.float16,    # medio-precisión para reducir VRAM
        max_memory = max_memory
    )
    pipe = pipeline(task=task,model=model,tokenizer=tokenizer,
       device_map="auto",            # idem
       torch_dtype=torch.float16,    # idem
       **kwargs
    )
    
    _loaded_models[name] = pipe
    _current_model = name
    return pipe
