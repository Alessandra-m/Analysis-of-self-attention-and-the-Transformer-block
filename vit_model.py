from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from transformer_encoder import MultiHeadAttention

RESULT_DIR = Path(__file__).resolve().parent / "result_ViT"
IMG_PATH = RESULT_DIR / "img.jpg"
PATCH_SIZE = 16


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class EncoderBlock(nn.Module):
    """Pre-LN encoder block used in ViT."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, weights = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, weights


class ViT(nn.Module):
    def __init__(self, img_h, img_w, patch_size=16, in_channels=3, d_model=192,
                 depth=6, num_heads=3, d_ff=768, num_classes=10):
        super().__init__()
        N = (img_h // patch_size) * (img_w // patch_size)

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                      # (B, N+1, D)
        x = x + self.pos_embed

        for blk in self.blocks:                             # z_1 … z_L
            x, weights = blk(x)

        x = self.norm(x)                                    # LN(z_L)
        y = x[:, 0]                                         # z_L⁰ — class token
        return self.head(y), weights


def main():
    torch.manual_seed(0)

    img = Image.open(IMG_PATH).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0)                      # (1, C, H, W)
    print("Input:               ", tuple(x.shape))

    H, W = x.shape[2] // PATCH_SIZE * PATCH_SIZE, x.shape[3] // PATCH_SIZE * PATCH_SIZE
    x = x[:, :, :H, :W]
    print("After crop:          ", tuple(x.shape))

    model = ViT(H, W, patch_size=PATCH_SIZE).eval()
    Gh, Gw = H // PATCH_SIZE, W // PATCH_SIZE

    with torch.no_grad():
        z = model.patch_embed(x).flatten(2).transpose(1, 2)
        print("Patch embedding:     ", tuple(z.shape))

        cls = model.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)
        print("+ class token:       ", tuple(z.shape))

        z = z + model.pos_embed
        print("+ pos embedding:     ", tuple(z.shape))

        for i, blk in enumerate(model.blocks, 1):
            z, weights = blk(z)
            print(f"EncoderBlock {i}:      ", tuple(z.shape))
        print("Attention weights:   ", tuple(weights.shape))     # (1, heads, N+1, N+1)

        z = model.norm(z)
        y = z[:, 0]
        print("y = LN(z_L⁰):        ", tuple(y.shape))           # (1, D)

        logits = model.head(y)
        print("Logits:              ", tuple(logits.shape))

        logits_ref, _ = model(x)
        print("Matches model(x):    ", torch.allclose(logits, logits_ref, atol=1e-6))
        print("Predicted class:     ", logits.argmax(-1).item(), "(random weights)")

    attn = weights[0, :, 0, 1:].mean(0).reshape(Gh, Gw)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(x[0].permute(1, 2, 0))
    axes[0].set_title("Input")
    axes[1].imshow(attn, cmap="viridis")
    axes[1].set_title("Attention [class] → patches")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "attention_map.jpg", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
