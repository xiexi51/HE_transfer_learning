import torch
import torch.nn as nn

class PolyKernelAttentionTimmCompat(nn.Module):
    """
    Drop-in replacement for timm.models.vision_transformer.Attention
    that removes softmax and uses polynomial-kernel attention:
        K = (alpha + beta * (QK^T / sqrt(d)))^p
        Y = K @ V / T   (T = seq_len, constant average to stabilize scale)
    Keeps qkv/proj/dropout shapes & semantics consistent with timm.
    """
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 degree=2, alpha=1.0, beta=1.0, average_by_len=True):
        super().__init__()
        assert degree in (2, 3), "Use low degree (2 or 3) for stability."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.average_by_len = average_by_len

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # B, N, 3C
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, h, N, dh
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, h, N, dh

        # similarity (linear ops only)
        s = (q @ k.transpose(-2, -1)) * self.scale  # B, h, N, N

        # polynomial kernel: (alpha + beta * s)^degree
        K = self.alpha + self.beta * s
        if self.degree == 2:
            K = K * K
        else:  # degree == 3
            K = K * K * K

        # simple constant averaging to control growth
        if self.average_by_len:
            K = K * (1.0 / N)

        K = self.attn_drop(K)  # keep dropout semantics

        y = K @ v  # B, h, N, dh
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


def replace_vit_with_poly_attention(model: nn.Module,
                                    degree=2, alpha=1.0, beta=1.0,
                                    average_by_len=True):
    """
    Only replace every block.attn with PolyKernelAttentionTimmCompat.
    NO changes to position embeddings / forward_features.
    """
    for blk in model.blocks:
        old_attn = blk.attn
        new_attn = PolyKernelAttentionTimmCompat(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=True,
            attn_drop=float(old_attn.attn_drop.p) if isinstance(old_attn.attn_drop, nn.Dropout) else 0.0,
            proj_drop=float(old_attn.proj_drop.p) if isinstance(old_attn.proj_drop, nn.Dropout) else 0.0,
            degree=degree, alpha=alpha, beta=beta, average_by_len=average_by_len
        )
        # optional init similar to timm
        nn.init.trunc_normal_(new_attn.qkv.weight, std=0.02)
        if new_attn.qkv.bias is not None:
            nn.init.zeros_(new_attn.qkv.bias)
        nn.init.trunc_normal_(new_attn.proj.weight, std=0.02)
        if new_attn.proj.bias is not None:
            nn.init.zeros_(new_attn.proj.bias)

        blk.attn = new_attn

    return model
