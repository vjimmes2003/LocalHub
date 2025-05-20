# spch_to_text/app.py

import os
import gradio as gr
from spch_to_text.model import (
    load_model, transcribe_audio, ensure_model_downloaded
)
from spch_to_text.utils.audio import delete_temp_files

state = {"model": None, "mode": None}

custom_css = open("spch_to_text/custom_style.css", "r", encoding="utf-8").read()

# Descarga modelos al iniciar
for key in ["turbo", "accurate"]:
    try:
        print(f"🔍 Verificando modelo: {key}")
        ensure_model_downloaded(key)
    except Exception as e:
        print(f"❌ Error al descargar el modelo '{key}': {e}")

def init_model(mode_choice):
    print(f"🚀 Inicializando modelo: {mode_choice}")
    try:
        model, actual = load_model(mode_choice)
        state.update({"model": model, "mode": actual})
        print(f"✅ Modelo cargado correctamente: {actual}")
        return f"✅ Modelo cargado: **{actual.upper()}**"
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return f"❌ Error al cargar modelo: {e}"

def process(audio_path, mode_choice, language_choice):
    
    import threading

    timeout_timer = threading.Timer(120, lambda: unload_model_audio())
    timeout_timer.start()

    if not audio_path:
        return "⚠️ Por favor sube un archivo de audio.", None, None
    print(f"📥 Audio recibido: {audio_path}")

    if state["model"] is None or state["mode"] != mode_choice:
        init_model(mode_choice)

    try:
        lang = None if language_choice == "Auto" else language_choice.lower()
        print(f"🌍 Idioma seleccionado: {lang or 'auto'}")
        full_text, timestamps = transcribe_audio(state["model"], audio_path, lang)

        srt_output = "\n".join([
            f"{i+1}\n{format_srt_time(t['start'])} --> {format_srt_time(t['end'])}\n{t['text']}\n"
            for i, t in enumerate(timestamps)
        ])

        txt_file = "output_transcription.txt"
        srt_file = "output_subtitles.srt"

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_output)

        timeout_timer.cancel()
        print(f"✅ Transcripción finalizada: {txt_file}, {srt_file}")
        return full_text, txt_file, srt_file
    except Exception as e:
        print(f"❌ Error durante transcripción: {e}")
        return f"❌ Error: {e}", None, None

def unload_model_audio():
    from spch_to_text.model import STATE
    if STATE["model"] is not None:
        print("⏱️ Audio: modelo superó los 2 minutos. Descargando de VRAM...")
        STATE["model"] = None
        STATE["mode"] = None
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()


def reset_app():
    print("🔁 Reseteando aplicación...")
    delete_temp_files()
    return None, None, None

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def create_whisper_interface():
    with gr.Blocks(title="🎤 Transcriptor de Audio", css=custom_css, theme = 'Taithrah/Minimal') as demo:
        gr.HTML("""<div style="width:100%;display:flex;justify-content:center;"><a href="/" style="text-decoration:none;padding:0.6em 1em;background:#f0f0f0;color:#333;border-radius:8px;font-weight:bold;border:1px solid #ccc;width:100%;font-size:25px;text-align:center;">🔙 Volver al inicio</a></div>""")

        gr.Markdown("""
### 🎧 Transcriptor de Audio con Whisper v3

Sube un archivo `.mp3` o `.wav`, selecciona el modelo y el idioma.  
Obtendrás el texto transcrito y los subtítulos listos para usar (.srt).

> 💡 **Whisper-Turbo** si tienes poca VRAM.  
> 🧠 **Whisper-Accurate** para máxima calidad (requiere > 6 GB VRAM).
""")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="🎙️ Archivo de Audio", type="filepath")
                example_selector = gr.Dropdown(
                    label="🗂️ Usar audio de ejemplo",
                    choices=[
                        "spch_to_text/assets/prueba_1_reloj.mp3",
                        "spch_to_text/assets/prueba_2_silencio.mp3",
                        "spch_to_text/assets/prueba_3_lucia.mp3"
                    ],
                    value=None
                )
                mode_choice = gr.Radio(
                    choices=["auto", "turbo", "accurate"],
                    value="auto",
                    label="⚙️ Modelo"
                )
                language_choice = gr.Dropdown(
                    choices=["Auto", "Spanish", "English", "French", "German", "Italian"],
                    value="Auto",
                    label="🌍 Idioma"
                )
                status = gr.Markdown("⏳ Modelo no cargado")
                unload_status = gr.Textbox(label="Estado del modelo", interactive=False)
                unload_btn = gr.Button("🔌 Descargar modelo de VRAM")

                def click_unload():
                    from spch_to_text.model import unload_model
                    success = unload_model()
                    return "✅ Modelo descargado" if success else "ℹ️ No había modelo cargado"

                unload_btn.click(click_unload, outputs=unload_status)

        example_selector.change(
            lambda p: p if os.path.exists(p) else None,
            inputs=[example_selector],
            outputs=[audio_input]
        )

        gr.Markdown("---")

        with gr.Row():
            transcribe_btn = gr.Button("📝 Transcribir", scale=1)
            reset_btn = gr.Button("🧹 Limpiar", scale=1)

        with gr.Row():
            output_text = gr.Textbox(label="📜 Transcripción", lines=14, interactive=False)

        with gr.Row():
            txt_download = gr.File(label="📄 Descargar TXT")
            srt_download = gr.File(label="🎞️ Descargar SRT")

        transcribe_btn.click(
            process,
            inputs=[audio_input, mode_choice, language_choice],
            outputs=[output_text, txt_download, srt_download]
        )

        reset_btn.click(
            reset_app,
            None,
            outputs=[output_text, txt_download, srt_download]
        )

        demo.load(lambda: init_model("auto"), None, status)

    return demo
