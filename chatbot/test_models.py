# File: chatbot/test_models.py

import os
import json
import gc
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from chatbot.config import MODEL_CONFIGS, CACHE_DIR
from chatbot.model import MODEL_PATHS

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SAVE_DIR = "chatbot/saved_chats"

INTERNAL_TO_FRIEND = { internal: friendly for friendly, internal in MODEL_PATHS.items()}

# 1) Definimos un SYSTEM_PROMPT general, y opcionalmente por modelo
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

TEST_QUESTIONS = {
    "Llama-3.2-3B-Instruct": [
        "¿Qué diferencia hay entre la inteligencia artificial débil y fuerte?",
        "Resume en una frase este texto: La inteligencia artificial es una disciplina que busca crear sistemas que puedan realizar tareas que requieren inteligencia humana.",
        "¿Cómo se dice “No tengo tiempo para esto” en japonés?",
        "¿Cuál es la capital de Mongolia y cuántos habitantes tiene?",
        "¿Podrías explicarlo como si tuviese 10 años?"
    ],
    "Instella-3B-Instruct": [
        "¿Qué pasos seguirías para preparar un sistema de recomendación?",
        "Si A implica B y B implica C, ¿qué se puede concluir sobre A y C?",
        "¿Qué errores ves en este pseudocódigo?\n\nif user = \"admin\":\n    print(\"Access granted\")\nelse:\n    print(\"Denied\")",
        "Redacta un correo profesional para pedir presupuesto a una empresa de servicios IT.",
        "Explica cómo resolverías un Sudoku paso a paso."
    ],
    "Qwen2.5-3B-Instruct": [
        "¿Cuáles son las ventajas y desventajas del teletrabajo?",
        "Dime cómo organizar un viaje a Japón en 5 pasos.",
        "¿Puedes explicarme el ciclo del agua usando subtítulos para cada fase?",
        "Hazme un esquema de los tipos de inteligencia según Howard Gardner.",
        "¿Qué necesito para montar un servidor web casero?"
    ],
    "Stable-Code-Instruct-3B": [
        "Corrige este error en Python: TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "¿Cómo harías una función en JavaScript para detectar si un número es primo?",
        "¿Qué hace este fragmento de código?\n\ndef f(x): return x if x == 0 else f(x - 1) + 1",
        "Genérame un script Bash para hacer backup de una carpeta cada día.",
        "Escribe una clase en Python con __init__, __str__ y un método update()."
    ]
}

def run_tests_and_save():
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"➡️ Directorio de ejemplos: '{SAVE_DIR}'\n")

    for alias, cfg in MODEL_CONFIGS.items():
        repo_id = cfg["repo_id"]
        example_path = os.path.join(SAVE_DIR, f"Ejemplo_{alias}.json")
        if os.path.exists(example_path):
            logger.info(f"✅ Ejemplo ya existe para ‘{alias}’, lo salto.\n")
            continue

        logger.info(f"🧠 Iniciando pruebas para modelo: {alias}")
        cache_dir = os.path.join(CACHE_DIR, alias)
        logger.info(f"  📁 Cache dir: {cache_dir}")

        # Preparar kwargs de generación
        gen_kwargs = cfg["pipeline_kwargs"].copy()
        gen_kwargs.pop("early_stopping", None)
        gen_kwargs.pop("no_repeat_ngram_size", None)
        gen_kwargs["do_sample"] = True
        logger.info(f"  ⚙️ Pipeline kwargs: {gen_kwargs}")

        # trust_remote_code si es necesario
        trust_remote = cfg.get("trust_remote_code", False) or "Instella" in alias
        if trust_remote:
            logger.warning("  ⚠️ trust_remote_code=True")

        # Calcular auto-asignación de memoria GPU (90%)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mib = props.total_memory // (1024**2)
            allow_mib = int(total_mib * 0.9)
            max_mem_arg = {0: f"{allow_mib}MiB"}
            logger.info(f"    🖥️ Asignando hasta {allow_mib}MiB en device_map")
        else:
            max_mem_arg = None

        # Cargar tokenizer y modelo
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote,
            device_map="auto",
            torch_dtype=torch.float16,
            **({"max_memory": max_mem_arg} if max_mem_arg else {})
        )
        elapsed = time.time() - start
        logger.info(f"  📦 Modelo cargado en {elapsed:.1f}s con device_map='auto'")

        # Construir pipeline
        pipe = pipeline(
            task=cfg["task"],
            model=model,
            tokenizer=tokenizer,
            **gen_kwargs
        )

        # Iterar preguntas
        history = []
        for i, question in enumerate(TEST_QUESTIONS[alias], start=1):
            # 2) Preparamos el prompt completo:
            system_prompt = PER_MODEL_PROMPT.get(alias, DEFAULT_SYSTEM_PROMPT)
            full_input = f"{system_prompt}\n\nUsuario: {question}\nAsistente:"
            
            logger.info(f"  📝 Pregunta {i}: {question!r}")
            q_start = time.time()
            try:
                out = pipe(full_input, return_full_text=False)[0]["generated_text"]
                # opcionalmente, si el modelo repite todo el prompt:
                if "Usuario:" in out:
                    out = out.split("Usuario:")[0].rstrip()
                dur = time.time() - q_start
                snippet = out.replace("\n", " ")[:200]
                logger.info(f"    💬 Respuesta ({dur:.1f}s): {snippet}{'...' if len(out)>200 else ''}")
            except Exception as e:
                dur = time.time() - q_start
                logger.error(f"    ❌ Error tras {dur:.1f}s: {e}")
                out = f"[ERROR] {e}"

            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant", "content": out})

            # Mostrar uso de GPU tras cada respuesta
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()  / 1024**2
                resv  = torch.cuda.memory_reserved()   / 1024**2
                logger.info(f"    🖥️ GPU alloc: {alloc:.1f}MiB, reserved: {resv:.1f}MiB")

        # Guardar historial de ejemplo
        friendly_name = INTERNAL_TO_FRIEND.get(alias, alias)
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump({"model": friendly_name, "history": history}, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Guardado como '{example_path}'")

        # Liberar memoria
        del pipe, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"🔻 Memoria vaciada tras prueba de ‘{alias}’\n")

if __name__ == "__main__":
    run_tests_and_save()
