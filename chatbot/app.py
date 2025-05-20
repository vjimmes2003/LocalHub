from transformers import TextIteratorStreamer
from threading import Thread
import os
import json
import gradio as gr
import time
import asyncio
from chatbot.model import load_model, MODEL_PATHS, unload_model
from chatbot.model_downloader import ensure_models_downloaded


# Forzar descarga si hace falta
ensure_models_downloaded()

#SYSTEM_PROMPT = "Eres un asistente que siempre es útil y ayuda con todo lo que sabe, pero a menos que te pidan ayuda con idiomas solo podrás responder en ESPAÑOL."

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente que siempre es útil y ayuda con todo lo que sabe, "
    "pero a menos que te pidan ayuda con idiomas solo podrás responder en ESPAÑOL."
)

PER_MODEL_PROMPT = {
    "Llama-3.2-3B-Instruct": (
        "Eres un experto en IA con un amplio conocimiento de los conceptos de IA débil y fuerte. "
        "Proporciona respuestas detalladas, usa ejemplos y clarifica por qué cada término recibe ese nombre."
    ),
    "Instella-3B-Instruct": (
        "Eres un asistente optimizado por AMD, excelente explicando procesos y cadenas de razonamiento "
        "de forma clara y fluida. Aprovecha tu fuerza en coherencia para estructurar bien las respuestas."
    ),
    "Qwen2.5-3B-Instruct": (
        "Eres un especialista en respuestas estructuradas. Ofrece esquemas, listas y subtítulos cuando sea útil."
    ),
    "Stable-Code-Instruct-3B": (
        "Eres un asistente de programación: explica paso a paso la corrección de errores y genera código limpio."
    ),
}

TEMP_CHAT_PATH = "chatbot/saved_chats/_temp.json"
CHAT_DIR = "chatbot/saved_chats"




def respond_stream(message, chat_history, model_choice):
    import threading

    timeout_timer = threading.Timer(120, lambda: unload_model_chatbot())

    def unload_model_chatbot():
        from chatbot.model import state
        if "pipe" in state:
            print("⏱️ Chatbot: modelo superó los 2 minutos. Descargando de VRAM...")
            del state["pipe"]
            del state["model_name"]
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()

    pipe = load_model(model_choice)
    tokenizer = pipe.tokenizer
    model = pipe.model

    internal = MODEL_PATHS.get(model_choice, model_choice)
    prompt = DEFAULT_SYSTEM_PROMPT
    if internal in PER_MODEL_PROMPT:
        prompt += "\n" + PER_MODEL_PROMPT[internal]
    prompt += "\n\n"
    chat_history = chat_history or []
    for msg in chat_history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos = tokenizer.eos_token_id
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        eos_token_id=eos,
        pad_token_id=eos
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})

    partial = ""
    last_yield = time.time()

    try:
        for token in streamer:
            partial += token
            clean = partial.split("User:")[0].split("Assistant:")[-1].strip()
            chat_history[-1]["content"] = clean

            # 🔐 Guardar en _temp.json
            with open(TEMP_CHAT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_choice,
                    "history": chat_history
                }, f, ensure_ascii=False, indent=2)

            # 🔁 Enviar actualizaciones al frontend
            if time.time() - last_yield > 0.5:
                yield None, [
                    {"role": m["role"], "content": m["content"]}
                    for m in chat_history
                ], chat_history
                
    except asyncio.CancelledError:
        #el cliente cerró la conexión; dejamos el generator
        return

    # última actualización
    yield None, [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history
    ], chat_history
    timeout_timer.cancel()

def save_chat(_, __, name):
    if not os.path.exists(TEMP_CHAT_PATH):
        return gr.update()
    with open(TEMP_CHAT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = name.strip()
    if not title:
        for m in data["history"]:
            if m["role"] == "user" and len(m["content"]) > 10:
                title = m["content"][:50].strip().replace(" ", "_").replace("?", "").replace(".", "")
                break
        else:
            title = "chat"
    fname = f"{title}.json"
    path = f"{CHAT_DIR}/{fname}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return gr.update(choices=get_chat_list())

def load_chat(fname):
    path = f"{CHAT_DIR}/{fname}"
    if not os.path.exists(path):
        return [], [], "Llama-3.2"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["history"], data["history"], data.get("model", "Llama-3.2")

def delete_chat(fname):
    path = f"{CHAT_DIR}/{fname}"
    if os.path.exists(path):
        os.remove(path)
    return gr.update(choices=get_chat_list())

def clear_chat():
    # Borra el archivo temporal
    if os.path.exists(TEMP_CHAT_PATH):
        os.remove(TEMP_CHAT_PATH)
    return "", [], [] # Limmpia input, historial visual y estado

def disable_input():
    return gr.update(interactive=False, placeholder="⏳ Esperando respuesta..."), gr.update(interactive=False)

def enable_input():
    return gr.update(interactive=True, placeholder="Escribe aquí"), gr.update(interactive=True)

def get_chat_list():
    return [f for f in sorted(os.listdir(CHAT_DIR), reverse=True) if f.endswith(".json") and not f.startswith("_")]

def create_chatbot_interface():
    custom_css = open("chatbot/custom_style.css", "r", encoding="utf-8").read()
    def get_chat_list():
        files = [f for f in os.listdir("chatbot/saved_chats") if f.endswith(".json") and not f.startswith("_")]
        return sorted(files, key=lambda f: os.path.getmtime(f"chatbot/saved_chats/{f}"), reverse=True)

    with gr.Blocks(css=custom_css, theme = 'Taithrah/Minimal') as demo:
        gr.HTML(
    """
    <div style="width: 100%; display:flex; flex-direction:row; flex: 1 1; align-items: center; justify-content: center; height:auto;">
        <a href="/" style="
            text-decoration: none;
            padding: 0.6em 1em;
            background-color: #f0f0f0;
            color: #333;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #ccc;
            display: block;
            width:100%;
            font-size:25px;
            text-align:center;
        ">🔙 Volver al inicio</a>
    </div>
    """)
        gr.Markdown("""
### 🤖 Chatbot Multi-Modelo Personalizable

Interactúa con modelos de lenguaje según tus necesidades. Guarda tus chats, cámbialos fácilmente desde la barra lateral y explora distintos comportamientos por modelo.

#### 🧠 Modelos disponibles:
- **🦙 Llama 3.2** – Modelo versátil de Meta, ideal para tareas generales, explicaciones multilingües y un equilibrio entre velocidad y precisión.
- **💡 Instella 3B Instruct** – Optimizado por AMD para comprensión de instrucciones, razonamiento y tareas conversacionales, ideal si buscas una respuesta fluida y coherente con recursos limitados.
- **🧾 Qwen2.5** – Especialista en respuestas estructuradas y seguimiento preciso de instrucciones. Muy útil en tareas analíticas y generación de textos complejos.
- **💻 Stable-Code 3B** – Focalizado en programación, ofrece respuestas útiles en generación de código, explicación de errores y completado de funciones.
""", elem_id="component-1")

        toggle_btn = gr.Button("🗨️", elem_id="chat-toggle-button")
        sidebar_state = gr.State(False)

        with gr.Row():
            with gr.Column(scale=0, elem_classes=["sidebar"]) as sidebar:
                #Para que funcione el selector ya que el último que selecciona como que deja de funcionar entonces poniendole un value random como el mismisimo info ya como que lo carga vacío y entonces podemos poner uno de los ejemplos.
                chat_selector = gr.Dropdown(choices=["Elije un chat a cargar..."] + get_chat_list(), label="📂 Chats guardados", value="Elije un chat a cargar...", info="Elije un chat a cargar...")
                load_btn = gr.Button("🔁 Cargar chat")
                delete_btn = gr.Button("🗑️ Eliminar chat")
                clear_btn = gr.Button("🧹 Nuevo chat")
            sidebar.visible = False

            with gr.Column(scale=4):
                model_choice = gr.Dropdown(
                    ["Llama-3.2", "Instella-3B", "Qwen2.5", "Stable-Code"],
                    label="🧠 Modelo activo",
                    value="Llama-3.2"
                )
                unload_status = gr.Textbox(label="Estado del modelo", interactive=False)

                unload_btn = gr.Button("🔌 Descargar modelo de VRAM")

                def click_unload():
                    success = unload_model()
                    if success:
                        return "✅ Modelo descargado de la VRAM"
                    else:
                        return "ℹ️ No hay modelo cargado"

                unload_btn.click(click_unload, outputs=unload_status)
                chatbot = gr.Chatbot(label="Chat", type="messages", show_copy_button=True,)
                chat_state = gr.State([])

                with gr.Row():
                    msg = gr.Textbox(label="Mensaje", placeholder="Escribe aquí", scale=4, lines=2, max_lines=10)
                    send = gr.Button("⬆️Enviar", scale=2)

                with gr.Row():
                    save_name = gr.Textbox(label="💾 Guardar como (opcional)", placeholder="Escriba aquí",lines=2,max_lines=10)
                    save_btn = gr.Button("💾Guardar chat",scale=2)

        # -- Funcionalidad general --
        send.click(
            disable_input,
            outputs=[msg, send]
        ).then(
            respond_stream,
            inputs=[msg, chat_state, model_choice],
            outputs=[msg, chatbot, chat_state]
        ).then(
            enable_input,
            outputs=[msg, send]
        )

        clear_btn.click(
            lambda: ("", [], []),  # limpia input, historial visual y estado
            None,
            [msg, chatbot, chat_state]
        ).then(
            lambda: gr.update(choices=get_chat_list()),
            None,
            [chat_selector]
        )

        def toggle_sidebar_fn(current_state):
            new_state = not current_state
            return (
                gr.update(visible=new_state),
                gr.update(value="❌" if new_state else "🗨️"),
                new_state
            )

        toggle_btn.click(
            toggle_sidebar_fn,
            [sidebar_state],
            [sidebar, toggle_btn, sidebar_state]
        ).then(
            lambda: gr.update(choices=get_chat_list()),
            None,
            [chat_selector]
        )
        
        def save_chat(_, __, name):
            try:
                if not os.path.exists(TEMP_CHAT_PATH):
                    return gr.update()
                with open(TEMP_CHAT_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = name.strip()
                if not title:
                    for m in data["history"]:
                        if m["role"] == "user" and len(m["content"]) > 10:
                            title = m["content"][:50].strip().replace(" ", "_").replace("?", "").replace(".", "")
                            break
                    else:
                        title = "chat"
                fname = f"{title}.json"
                path = f"chatbot/saved_chats/{fname}"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return gr.update(choices=get_chat_list())
            except Exception as e:
                print("❌ Error al guardar:", e)
                return gr.update()

        save_btn.click(save_chat, [chatbot, model_choice, save_name], chat_selector)

        def load_chat(fname):
            path = f"chatbot/saved_chats/{fname}"
            if not os.path.exists(path):
                return [], [], "Llama-3.2"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data["history"]
            model = data.get("model", "Llama-3.2")
            return history, history, model

        def delete_chat(fname):
            try:
                os.remove(f"chatbot/saved_chats/{fname}")
            except:
                pass
            return gr.update(choices=get_chat_list())

        # -- Botones laterales --
        load_btn.click(load_chat, [chat_selector], [chatbot, chat_state, model_choice])
        delete_btn.click(delete_chat, [chat_selector], chat_selector)
        chat_selector.change(lambda: gr.update(choices=get_chat_list()), None, chat_selector)

    
    return demo