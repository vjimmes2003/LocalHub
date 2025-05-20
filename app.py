# app.py (FastAPI version)
import os, sys
from huggingface_hub import HfFolder

# Intentamos leer el token (CLI login o variable de entorno)
hf_token = HfFolder.get_token() or os.getenv("HF_HUB_TOKEN")

if not hf_token:
    print(
        "\n⛔️  Error: No Hugging Face token encontrado.\n"
        "  • Asegúrate de haber ejecutado `huggingface-cli login`, O\n"
        "  • Define la variable de entorno HF_HUB_TOKEN con tu token.\n"
        "  Luego vuelve a arrancar la aplicación.\n"
    )
    sys.exit(1)

# Opcional: inyectar el token en el entorno para que Transformers/HuggingFace lo use
os.environ["HF_HUB_TOKEN"] = hf_token

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import gradio as gr

from chatbot.app import create_chatbot_interface
from images.app import create_image_interface
from spch_to_text.app import create_whisper_interface


app = FastAPI()


# Monta la carpeta static en /static
app.mount("/static", StaticFiles(directory="static"), name="static")


# Monta tus apps de Gradio…
gr.mount_gradio_app(app, create_chatbot_interface(), path="/chatbot")
gr.mount_gradio_app(app, create_image_interface(),   path="/images")
gr.mount_gradio_app(app, create_whisper_interface(),  path="/whisper")

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("templates/index.html", "r", encoding="utf-8").read()
