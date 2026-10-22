import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        out, weights = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        return out, weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class EncoderBlock(nn.Module):
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
    def __init__(self, img_h, img_w, patch=16, in_ch=3, D=192,
                 depth=6, num_heads=3, d_ff=768, num_classes=10):
        super().__init__()
        N = (img_h // patch) * (img_w // patch)

        self.patch_embed = nn.Conv2d(in_ch, D, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, D)) 
        self.blocks = nn.ModuleList([EncoderBlock(D, num_heads, d_ff) for _ in range(depth)])
        self.norm = nn.LayerNorm(D) # LN: y = LN(z_L⁰)              
        self.head = nn.Linear(D, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, D)
        x = x + self.pos_embed

        for blk in self.blocks:                              # z_1 … z_L
            x, weights = blk(x)

        x = self.norm(x)                                     # LN(z_L)
        y = x[:, 0]                                          # z_L⁰ — class-токен
        return self.head(y), weights


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


def main():
    torch.manual_seed(0)
    PATCH = 16

    img = Image.open("img.jpg").convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0)                # (1, 3, H, W)
    print("Исходный вход:        ", tuple(x.shape))
    H, W = x.shape[2] // PATCH * PATCH, x.shape[3] // PATCH * PATCH
    x = x[:, :, :H, :W]
    print("После обрезки:        ", tuple(x.shape))

    model = ViT(H, W, patch=PATCH).eval()
    Gh, Gw = H // PATCH, W // PATCH

    with torch.no_grad():
        z = model.patch_embed(x).flatten(2).transpose(1, 2)
        print("Patch embedding:      ", tuple(z.shape))   

        cls = model.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)
        print("+ class token:        ", tuple(z.shape))  

        z = z + model.pos_embed
        print("+ pos embedding:      ", tuple(z.shape))

        for i, blk in enumerate(model.blocks, 1):
            z, weights = blk(z)
            print(f"EncoderBlock {i}:       ", tuple(z.shape))
        print("Веса внимания:        ", tuple(weights.shape))  # (1, heads, N+1, N+1)

        z = model.norm(z)
        y = z[:, 0]
        print("y = LN(z_L⁰):         ", tuple(y.shape))   # (1, D)

        logits = model.head(y)
        print("Логиты:               ", tuple(logits.shape))

        logits_ref, _ = model(x)
        print("Совпадает с model(x): ", torch.allclose(logits, logits_ref, atol=1e-6))
        print("Предсказанный класс:  ", logits.argmax(-1).item(), "(веса случайные)")


    attn = weights[0, :, 0, 1:].mean(0).reshape(Gh, Gw)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(x[0].permute(1, 2, 0))
    axes[0].set_title("Вход")
    axes[1].imshow(attn, cmap="viridis")
    axes[1].set_title("Внимание [class] → патчи")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("result_ViT/attention_map.png", dpi=120)
    plt.show()



if __name__ == "__main__":
    main()
