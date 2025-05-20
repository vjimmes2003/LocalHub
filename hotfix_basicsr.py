import os
import site

def apply_basicsr_patch():
    print("🔍 Buscando instalación de basicsr...")

    basicsr_path = None
    for path in site.getsitepackages():
        candidate = os.path.join(path, "basicsr", "data", "degradations.py")
        if os.path.exists(candidate):
            basicsr_path = candidate
            break

    if not basicsr_path:
        print("❌ No se encontró la instalación de basicsr.")
        return

    print(f"🛠️ Encontrado: {basicsr_path}")

    with open(basicsr_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "functional_tensor" not in content:
        print("✅ El archivo ya está parcheado. No se necesita hacer nada.")
        return

    content_fixed = content.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale"
    )

    with open(basicsr_path, "w", encoding="utf-8") as f:
        f.write(content_fixed)

    print("✅ Hotfix aplicado correctamente a basicsr.")

if __name__ == "__main__":
    apply_basicsr_patch()
