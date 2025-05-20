from pydub import AudioSegment
import os

def to_mp3(input_path: str, output_path: str):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3")
    return output_path

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_wav16k_mono(input_path: str, output_path: str):
    """
    Carga cualquier archivo de audio y lo convierte a WAV 16 kHz mono.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path
