import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
class SynapseAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx, weights=None):
        q = self.to_q(qx)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(kx).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if weights is not None:
            b, n = qx.shape[0], qx.shape[1]
            patch_size = int(math.sqrt(n))
            if patch_size * patch_size != n:
                patch_size = n

            if hasattr(weights, 'unsqueeze'):
                weights_flat = weights.unsqueeze(1) if weights.dim() == 3 else weights
                attn = self.attend(dots)
            else:
                attn = self.attend(dots)
        else:
            attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
