from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

RESULT_DIR = Path(__file__).resolve().parent / "result_ViT"
IMG_PATH = RESULT_DIR / "img.jpg"
PATCH_SIZE = 16
D = 768
C = 3


def main():
    img = Image.open(IMG_PATH).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0)                     # (1, C, H, W)
    print("Input:               ", tuple(x.shape))

    B, _, H, W = x.shape
    H, W = H // PATCH_SIZE * PATCH_SIZE, W // PATCH_SIZE * PATCH_SIZE
    x = x[:, :, :H, :W]
    print("After crop:          ", tuple(x.shape))

    Gh, Gw = H // PATCH_SIZE, W // PATCH_SIZE
    N = Gh * Gw

    patches = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.permute(0, 2, 3, 1, 4, 5)            # (B, H/P, W/P, C, P, P)
    patches = patches.reshape(B, N, C, PATCH_SIZE, PATCH_SIZE)  # (B, N, C, P, P)
    print("Patches:             ", tuple(patches.shape))

    fig, axes = plt.subplots(Gh, Gw, figsize=(Gw * 0.6, Gh * 0.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(patches[0, i].permute(1, 2, 0).numpy())  # (P, P, C)
        ax.axis("off")
    plt.suptitle(f"{N} patches {PATCH_SIZE}x{PATCH_SIZE} (grid {Gh}x{Gw})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "patches_grid.jpg", dpi=120)
    plt.show()

    patch_embed = nn.Conv2d(C, D, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    z = patch_embed(x)                                     # (B, D, H/P, W/P)
    z = z.flatten(2)                                       # (B, D, N)
    z = z.transpose(1, 2)                                  # (B, N, D)
    print("Patch embeddings:    ", tuple(z.shape))

    E = patch_embed.weight.reshape(D, -1).T                # (P*P*C, D)
    z_ref = patches.reshape(B, N, -1) @ E + patch_embed.bias
    print("Conv2d == flatten @ E:", torch.allclose(z, z_ref, atol=1e-5))


if __name__ == "__main__":
    main()
