import gradio as gr
import os
import glob
from .model import generate_image
from images.utils.config import MODEL_CONFIGS
from images.utils.bootstrap import bootstrap_all

# Desactiva avisos de symlinks en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

bootstrap_all()

preset_options = {
    key: [f"{w}x{h}" for (w, h) in config["available_resolutions"]]
    for key, config in MODEL_CONFIGS.items()
}

def mostrar_imagen(nombre, tipo):
    return f"images/examples/{nombre}_{tipo}.png"

ejemplos_realistic = [f"realisticvision-v6_{i}" for i in range(1, 6)]
ejemplos_juggernaut = [f"juggernautxl_{i}" for i in range(1, 6)]

PROMPTS_REALISTIC = [
    "instagram photo, closeup face photo of 23 y.o Chloe in black sweater, cleavage, pale skin, (smile:0.4), hard shadows",
    "closeup face photo of caucasian man in black clothes, night city street, bokeh",
    "instagram photo, front shot, portrait photo of a 24 y.o woman, wearing dress, beautiful face, cinematic shot, dark shot",
    "oil painting, underwater image of an ancient city, glowing, by Greg Rutkowski",
    "polaroid photo of a road, warm tones, perfect landscape"
]

PROMPTS_JUGGERNAUT = [
    "beautiful lady, (freckles), big smile, ruby eyes, long curly hair, dark makeup, hyperdetailed photography, soft light, head and shoulders portrait, cover",
    "Leica portrait of a gremlin skateboarding, coded patterns, sparse and simple, uhd image, urbancore, sovietwave, period snapshot",
    "A hyperdetailed photograph of a Cat dressed as a mafia boss holding a fish walking down a Japanese fish market with an angry face, 8k resolution, best quality, beautiful photograph, dynamic lighting",
    "photograph, a path in the woods with leaves and the sun shining, by Julian Allen, dramatic autumn landscape, rich cold moody colours, hi resolution",
    "a torso with a TV instead of a head"
]
custom_css = open("images/custom_style.css", "r", encoding="utf-8").read()

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
    with gr.Accordion("ℹ️ Cómo usar la app (haz clic para abrir/cerrar)", open=False):
        gr.Markdown("""
### ✨ Generador de imágenes por texto (txt2img)

1. Escribe un *prompt* en inglés.  
2. Selecciona el modelo deseado (por defecto: **realisticvision-v6**).  
3. Elige una resolución recomendada.  
4. Pulsa en **🎨 Generar Imagen**.  
5. Espera unos segundos a que se procese. Verás **2 versiones**:
- Imagen generada normal  
- Imagen con mejoras (hires fix y/o upscale si está activado)  

💡 Cuanto más grande sea la imagen, más tardará en generarse.
""")

        gr.Markdown("### 🖼️ Ejemplos visuales — Realistic Vision V6 B1")
        for i, nombre in enumerate(ejemplos_realistic):
            with gr.Row():
                img_display = gr.Image(value=f"images/examples/{nombre}_base.png", interactive=False, show_label=False, width=300)
                with gr.Column():
                    gr.Markdown(f"📝 **Prompt:** `{PROMPTS_REALISTIC[i]}`")
                    btn1 = gr.Button("👁️ Ver ANTES")
                    btn2 = gr.Button("🎨 Ver DESPUÉS")
            btn1.click(fn=mostrar_imagen, inputs=[gr.State(nombre), gr.State("base")], outputs=img_display)
            btn2.click(fn=mostrar_imagen, inputs=[gr.State(nombre), gr.State("final")], outputs=img_display)

        gr.Markdown("### 🖼️ Ejemplos visuales — Juggernaut XL v9")
        for i, nombre in enumerate(ejemplos_juggernaut):
            with gr.Row():
                img_display = gr.Image(value=f"images/examples/{nombre}_base.png", interactive=False, show_label=False, width=300)
                with gr.Column():
                    gr.Markdown(f"📝 **Prompt:** `{PROMPTS_JUGGERNAUT[i]}`")
                    btn1 = gr.Button("👁️ Ver ANTES")
                    btn2 = gr.Button("🎨 Ver DESPUÉS")
            btn1.click(fn=mostrar_imagen, inputs=[gr.State(nombre), gr.State("base")], outputs=img_display)
            btn2.click(fn=mostrar_imagen, inputs=[gr.State(nombre), gr.State("final")], outputs=img_display)

    prompt = gr.Textbox(label="Prompt (en inglés)", placeholder="Ej: a cyberpunk city at sunset")

    with gr.Row():
        model_select = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            label="Modelo",
            value="realisticvision-v6"
        )

        resolution_radio = gr.Radio(
            choices=preset_options["realisticvision-v6"],
            label="Resolución recomendada",
            value=preset_options["realisticvision-v6"][1],
        )

    seed_input = gr.Number(label="Seed (opcional, -1 = aleatoria)", value=-1, precision=0)
    generate_btn = gr.Button("🎨 Generar Imagen")
    upscale_select = gr.Dropdown(
        label="Upscaler (opcional)",
        choices=["none", "realistic", "anime", "general"],
        value="none"
    )

    output_base = gr.Image(label="Imagen Base (sin Hires.Fix)", interactive=False, format="png")
    output_final = gr.Image(label="Imagen Final (con Hires.Fix y Upscaler si aplica)", interactive=False, format="png")

    def update_resolution_options(model_key):
        resolutions = preset_options[model_key]
        return gr.update(choices=resolutions, value=resolutions[0])

    def on_generate(prompt, model_key, resolution_str, seed, upscale_mode):
        width, height = map(int, resolution_str.split("x"))
        resolution = (width, height)
        seed = None if seed == -1 else int(seed)
        image_base, image_final = generate_image(prompt, model_key, resolution, seed, upscaler_key=upscale_mode)
        return image_base, image_final

    model_select.change(fn=update_resolution_options, inputs=model_select, outputs=resolution_radio)

    with gr.Row():
        gr.Markdown("## 🖼️ Comparación de imágenes generadas")

    with gr.Row():
        gr.Markdown("### 🎨 Imagen base (antes de mejora)")
        gr.Markdown("### 🚀 Imagen final (con hires.fix o upscale)")

    with gr.Row():
        gallery_base = gr.Gallery(
            label="🖼️ Imagenes Base",
            interactive=False,
            allow_preview=True,
            preview=True,
            show_label=False,
            columns=[4],
        )

        gallery_final = gr.Gallery(
            label="🖼️ Imagenes Final",
            interactive=False,
            allow_preview=True,
            preview=True,
            show_label=False,
            columns=[4],
        )

    def load_galleries():
        base = sorted(glob.glob("images/outputs/*_base.png"), key=os.path.getmtime, reverse=True)
        final = sorted(glob.glob("images/outputs/*_final.png"), key=os.path.getmtime, reverse=True)
        return base, final

    refresh_btn = gr.Button("🔄 Actualizar Galerías")

    refresh_btn.click(fn=load_galleries, inputs=[], outputs=[gallery_base, gallery_final])

    generate_btn.click(
        fn=on_generate,
        inputs=[prompt, model_select, resolution_radio, seed_input, upscale_select],
        outputs=[output_base, output_final]
    ).then(
        fn=load_galleries,
        inputs=[],
        outputs=[gallery_base, gallery_final]
    )


def create_image_interface():
    return demo
