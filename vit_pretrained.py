from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16

RESULT_DIR = Path(__file__).resolve().parent / "result_ViT"
IMG_PATH = RESULT_DIR / "img.jpg"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).eval()
    preprocess = weights.transforms()

    print("Modules:")
    print("  conv_proj:            ", model.conv_proj, f"{count_params(model.conv_proj):,} params")
    print("  class_token:          ", tuple(model.class_token.shape))
    print("  encoder.pos_embedding:", tuple(model.encoder.pos_embedding.shape))
    print("  encoder.layers:       ", len(model.encoder.layers), "x EncoderBlock,",
          f"{count_params(model.encoder.layers[0]):,} params each")
    print(model.encoder.layers[0])
    print("  encoder.ln:           ", model.encoder.ln)
    print("  heads:                ", model.heads.head)
    print("  Total params:         ", f"{count_params(model):,}")
    print()

    img = Image.open(IMG_PATH).convert("RGB")
    x = preprocess(img).unsqueeze(0)                        # (1, 3, 224, 224)
    print("Input after transforms: ", tuple(x.shape))

    with torch.no_grad():
        z = model.conv_proj(x).flatten(2).transpose(1, 2)   # (1, 196, 768)
        print("Patch embedding:        ", tuple(z.shape))

        cls = model.class_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)                      # (1, 197, 768)
        print("+ class token:          ", tuple(z.shape))

        z = z + model.encoder.pos_embedding
        print("+ pos embedding:        ", tuple(z.shape))

        for blk in model.encoder.layers[:-1]:
            z = blk(z)
        last = model.encoder.layers[-1]
        h = last.ln_1(z)
        attn_out, weights_attn = last.self_attention(h, h, h, need_weights=True)
        z = z + attn_out
        z = z + last.mlp(last.ln_2(z))
        print("After 12 blocks:        ", tuple(z.shape))
        print("Attention weights (last):", tuple(weights_attn.shape))  # (1, 197, 197), mean over heads

        y = model.encoder.ln(z)[:, 0]                       # (1, 768)
        logits = model.heads(y)
        print("Logits:                 ", tuple(logits.shape))
        print("Matches model(x):       ", torch.allclose(logits, model(x), atol=1e-4))

    probs = logits.softmax(-1)[0]
    top5 = probs.topk(5)
    categories = weights.meta["categories"]
    print()
    print("Top-5 ImageNet predictions:")
    for p, i in zip(top5.values, top5.indices):
        print(f"  {categories[i]:<25} {p.item():.3f}")

    attn = weights_attn[0, 0, 1:].reshape(14, 14)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mean = torch.tensor(preprocess.mean).view(3, 1, 1)
    std = torch.tensor(preprocess.std).view(3, 1, 1)
    axes[0].imshow((x[0] * std + mean).clamp(0, 1).permute(1, 2, 0))
    axes[0].set_title(f"Input 224x224: {categories[top5.indices[0]]}")
    axes[1].imshow(attn, cmap="viridis")
    axes[1].set_title("Pretrained ViT-B/16: attention [class] → patches")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "pretrained_attention_map.jpg", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
