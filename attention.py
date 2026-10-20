import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights

torch.manual_seed(0)
B, H, T, d_k, d_v = 2, 4, 5, 16, 16
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_v)

mine, _ = scaled_dot_product_attention(Q, K, V)
ref = F.scaled_dot_product_attention(Q, K, V)

print(torch.allclose(mine, ref, atol=1e-6))   # True
print((mine - ref).abs().max())                # ~1e-7