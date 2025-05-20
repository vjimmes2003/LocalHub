import torch

def analizar_pth(path):
    print(f"📦 Analizando: {path}")
    try:
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "params" in state_dict:
            state_dict = state_dict["params"]

        keys = list(state_dict.keys())
        print(f"🔑 Claves principales:")
        for k in keys[:10]:
            print(f" - {k}")

        if "conv_first.weight" in state_dict:
            print(f"✔️ Tiene conv_first.weight → probablemente RRDBNet")
            print(f"   Tamaño: {state_dict['conv_first.weight'].shape}")
        elif "body.0.weight" in state_dict:
            print(f"✔️ Tiene body.0.weight → probablemente SRVGGNetCompact")
            print(f"   Tamaño: {state_dict['body.0.weight'].shape}")
        else:
            print(f"⚠️ Estructura desconocida, revisar más claves...")

    except Exception as e:
        print(f"❌ Error cargando {path}: {e}")

# 👉 Cambia la ruta aquí por el archivo que quieras comprobar
analizar_pth("RealESRGAN_x4plus_anime_6B.pth")
analizar_pth("realesr-general-x4v3.pth")
analizar_pth("RealESRGAN_x4plus.pth")
