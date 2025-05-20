# spch_to_text/model.py

import os
import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from spch_to_text.utils.audio import to_wav16k_mono

# Configuración de modelos
MODELS = {
    "turbo": {
        "name": "faster-whisper-large-v3-turbo",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "vram": 6 * 1024**3
    },
    "accurate": {
        "name": "faster-whisper-large-v3",
        "repo": "Systran/faster-whisper-large-v3",
        "vram": 10 * 1024**3
    }
}

# Carpeta de almacenamiento
MODEL_DIR = "spch_to_text/models"

def ensure_model_downloaded(model_key: str):
    """Descarga manual del modelo completo desde Hugging Face."""
    model_data = MODELS[model_key]
    model_name = MODELS[model_key]["name"]
    local_path = os.path.join(MODEL_DIR, model_data["name"])

    if not os.path.exists(local_path):
        print(f"📥 Descargando modelo '{model_data['name']}' desde HuggingFace...")
        snapshot_download(
            repo_id=model_data["repo"],
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Modelo descargado en: {local_path}")
    else:
        print(f"✅ Modelo ya descargado: {model_name}")
        
    return local_path

def detect_mode(requested="auto"):
    if requested in MODELS:
        return requested
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        return "accurate" if total >= MODELS["accurate"]["vram"] else "turbo"
    return "turbo"

def load_model(requested_mode):
    """Carga el modelo con ruta local."""
    mode = detect_mode(requested_mode)
    model_path = ensure_model_downloaded(mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"🚀 Cargando modelo '{mode}' en {device.upper()} ({compute_type}) desde: {model_path}")

    model = WhisperModel(
        model_size_or_path=model_path,
        device=device,
        compute_type=compute_type
    )

    return model, mode

def transcribe_audio(model, audio_path, language=None):
    """Convierte a WAV, transcribe y devuelve texto + timestamps."""
    os.makedirs("spch_to_text/temp", exist_ok=True)
    wav_path = os.path.join("spch_to_text/temp", "input.wav")
    to_wav16k_mono(audio_path, wav_path)

    segments, info = model.transcribe(wav_path, beam_size=5, language=language)
    print(f"⏱️  Duración: {info.duration:.2f} segundos")

    full_text = ""
    timestamps = []
    for segment in segments:
        full_text += segment.text.strip() + " "
        timestamps.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return full_text.strip(), timestamps
