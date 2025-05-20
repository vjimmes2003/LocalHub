import torch

ckpt = torch.load("RealESRGAN_x4plus_anime_6B.pth", map_location="cpu")
state = ckpt.get("params_ema", ckpt.get("params", ckpt))

for key in list(state.keys())[:50]:
    print(key)

