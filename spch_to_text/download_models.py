# spch_to_text/download_models.py

import os
from spch_to_text.model import ensure_model_downloaded

def main():
    print("🔍 Verificando disponibilidad de modelos...")

    for model_key in ["turbo", "accurate"]:
        try:
            path = ensure_model_downloaded(model_key)
            print(f"✅ Modelo '{model_key}' disponible en: {path}")
        except Exception as e:
            print(f"❌ Error al descargar el modelo '{model_key}': {e}")

    print("\n🚀 ¡Todos los modelos necesarios están preparados!")

if __name__ == "__main__":
    main()
