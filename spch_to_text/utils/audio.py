# spch_to_text/utils/audio.py
import os
from pydub import AudioSegment

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def to_wav16k_mono(input_path, output_path):
    """
    Convierte cualquier audio a WAV mono 16 kHz para usar con Whisper
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

def delete_temp_files():
    """
    Limpia los archivos temporales usados en transcripción
    """
    if os.path.exists("spch_to_text/temp"):
        for f in os.listdir("spch_to_text/temp"):
            try:
                os.remove(os.path.join("spch_to_text/temp", f))
            except Exception:
                pass
