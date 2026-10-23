import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


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
        self.act = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class EncoderBlock(nn.Module):
    """Post-LN encoder block from "Attention Is All You Need"."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, weights = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, weights


def encoder_flops(T, d_model, d_ff):
    """FLOPs of one encoder block (B=1), matmuls only: 1 MAC = 2 FLOPs."""
    linear = 2 * (4 * T * d_model ** 2 + 2 * T * d_model * d_ff)  # W_q, W_k, W_v, W_o + FFN
    attn = 2 * (2 * T ** 2 * d_model)                             # Q·Kᵀ and weights·V
    return linear, attn


def main():
    torch.manual_seed(0)
    B, H, T, d_k, d_v = 2, 4, 5, 16, 16
    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_v)

    mine, _ = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    print("Matches F.scaled_dot_product_attention:", torch.allclose(mine, ref, atol=1e-6))
    print("Max abs difference:                    ", (mine - ref).abs().max().item())

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))     # True = attend
    mine, _ = scaled_dot_product_attention(Q, K, V, mask=causal)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal)
    print("Matches with causal mask:              ", torch.allclose(mine, ref, atol=1e-6))

    torch.manual_seed(0)
    d_model, num_heads = 32, 4
    mha = MultiHeadAttention(d_model, num_heads).eval()
    ref_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True).eval()
    with torch.no_grad():
        ref_mha.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref_mha.in_proj_bias.copy_(torch.cat([mha.W_q.bias, mha.W_k.bias, mha.W_v.bias]))
        ref_mha.out_proj.weight.copy_(mha.W_o.weight)
        ref_mha.out_proj.bias.copy_(mha.W_o.bias)

        x = torch.randn(B, T, d_model)
        out, weights = mha(x)
        ref_out, ref_weights = ref_mha(x, x, x, average_attn_weights=False)
    print("MHA matches nn.MultiheadAttention:     ",
          torch.allclose(out, ref_out, atol=1e-6) and torch.allclose(weights, ref_weights, atol=1e-6))

    torch.manual_seed(0)
    B, T, d_model, num_heads, d_ff = 2, 5, 32, 4, 64
    block = EncoderBlock(d_model, num_heads, d_ff).eval()

    x = torch.randn(B, T, d_model)
    out, weights = block(x)
    print("EncoderBlock output:                   ", tuple(out.shape))      # (B, T, d_model)
    print("Attention weights:                     ", tuple(weights.shape))  # (B, num_heads, T, T)

    encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(6)]).eval()
    for blk in encoder:
        x, _ = blk(x)
    print("Stack of 6 blocks output:              ", tuple(x.shape))        # (B, T, d_model)

    # Cost of one block of the paper's base model vs. number of tokens
    d_model, num_heads, d_ff = 512, 8, 2048
    block = EncoderBlock(d_model, num_heads, d_ff).eval()
    print()
    print(f"{'T':>6} {'Linear, GFLOPs':>15} {'Attn, GFLOPs':>13} {'Attn share':>11} "
          f"{'Measured':>9} {'Scores, MB':>11}")
    for T in [128, 512, 1024, 2048, 3072, 4096, 8192]:
        linear, attn = encoder_flops(T, d_model, d_ff)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            block(torch.randn(1, T, d_model))
        measured = counter.get_total_flops() == linear + attn
        scores_mb = num_heads * T ** 2 * 4 / 1e6                     # float32, B=1
        print(f"{T:>6} {linear / 1e9:>15.1f} {attn / 1e9:>13.1f} {attn / (linear + attn):>10.0%} "
              f"{str(measured):>9} {scores_mb:>11.1f}")


if __name__ == "__main__":
    main()
