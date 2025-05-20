# File: config.py

import os
from typing import Dict, Any

# Configuración de cada modelo: repo de HF y parámetros del pipeline
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Llama-3.2-3B-Instruct": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "task": "text-generation",
        "pipeline_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    },
    "Instella-3B-Instruct": {
        "repo_id": "amd/Instella-3B-Instruct",
        "task": "text-generation",
        "pipeline_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
        },
    },
    "Qwen2.5-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "task": "text-generation",
        "pipeline_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.85,
        },
    },
    "Stable-Code-Instruct-3B": {
        "repo_id": "stabilityai/stable-code-instruct-3b",
        "task": "text-generation",
        "pipeline_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
        },
    },
}

# Carpeta donde se guardarán los modelos descargados
CACHE_DIR = os.path.abspath("./chatbot/models")

def default_model() -> str:
    return "Llama-3.2-3B-Instruct"