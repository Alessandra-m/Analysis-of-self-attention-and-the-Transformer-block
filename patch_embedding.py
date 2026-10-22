import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

IMG_PATH = "result_ViT/img.jpg"
PATCH = 16               # размер патча P
D = 768              # размерность эмбеддинга (как в ViT-B)
C = 3                # каналы (RGB)


img = Image.open(IMG_PATH).convert("RGB")

transform = T.Compose([
    T.ToTensor(),                     # -> (C, H, W), значения [0,1], без ресайза
])
x = transform(img).unsqueeze(0)       # добавляем батч -> (1, C, H, W)
print("Исходный вход:", x.shape)


B, _, H, W = x.shape
H, W = H // PATCH * PATCH, W // PATCH * PATCH
x = x[:, :, :H, :W]
assert H % PATCH == 0 and W % PATCH == 0
print("Вход после обрезки:", x.shape)

Gh, Gw = H // PATCH, W // PATCH       # сетка патчей
N = Gh * Gw                           # число патчей


patches = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
patches = patches.permute(0, 2, 3, 1, 4, 5)   # (B, H/P, W/P, C, P, P)
patches = patches.reshape(B, N, C, PATCH, PATCH)  # (B, N, C, P, P)
print("Патчи (для визуализации):", patches.shape)


fig, axes = plt.subplots(Gh, Gw, figsize=(Gw * 0.6, Gh * 0.6))
for i, ax in enumerate(axes.flat):
    p = patches[0, i].permute(1, 2, 0).numpy()  # (P, P, C)
    ax.imshow(p)
    ax.axis("off")
plt.suptitle(f"{N} патчей {PATCH}x{PATCH} (сетка {Gh}x{Gw})")
plt.tight_layout()
plt.savefig("result_ViT/patches_grid.png", dpi=120)
plt.show()


patch_embed = nn.Conv2d(in_channels=C, out_channels=D,
                        kernel_size=PATCH, stride=PATCH)

z = patch_embed(x)              # (B, D, H/P, W/P)
z = z.flatten(2)                # (B, D, N)
z = z.transpose(1, 2)           # (B, N, D)
print("Эмбеддинги патчей:", z.shape)
