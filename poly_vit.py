import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utilities (RoPE)
# =========================
def apply_rope(x, cos, sin):
    """
    x:   [B,H,N,Dh] (Dh must be even)
    cos: [1,1,N,Dh] or [N,Dh] broadcastable
    sin: [1,1,N,Dh] or [N,Dh] broadcastable
    """
    Dh = x.size(-1)
    assert Dh % 2 == 0, "RoPE requires even head dim"
    x_2 = x.reshape(*x.shape[:-1], Dh // 2, 2)
    x_even = x_2[..., 0]
    x_odd  = x_2[..., 1]
    y_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
    y_odd  = x_even * sin[..., ::2] + x_odd * cos[..., ::2]
    y = torch.stack([y_even, y_odd], dim=-1).reshape_as(x)
    return y


# =========================
# Attention Variants
# =========================
class PolyKernelAttentionTimmCompat(nn.Module):
    """
    Drop-in replacement for timm Attention using polynomial-kernel attention:
        X = (QK^T) / sqrt(d)
        K = (alpha + beta * X)^p, with p in {2,3}
        Y = (K @ V) / T, T = seq_len
    Keeps qkv/proj/dropout semantics compatible with timm.
    """
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 degree=2, alpha=1.0, beta=1.0, average_by_len=True):
        super().__init__()
        assert degree in (2, 3), "Use low degree (2 or 3) for stability."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.degree = degree
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta  = nn.Parameter(torch.tensor(float(beta)))
        self.average_by_len = average_by_len

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Dh)
        q, k, v = qkv.unbind(dim=2)  # [B,N,H,Dh] each
        q = q.transpose(1, 2)        # [B,H,N,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        X = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N,N]
        K = (self.alpha + self.beta * X)
        K = K * K if self.degree == 2 else K * K * K           # degree 2 or 3
        if self.average_by_len:
            K = K / float(N)

        K = self.attn_drop(K)
        y = torch.matmul(K, v)                                  # [B,H,N,Dh]
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.proj_drop(self.proj(y))
        return y


class LinearPositionalAttention(nn.Module):
    """
    Linear attention + positional transform:
      Y = pos(Q) @ (pos(K)^T @ V) / N
    pos_type ∈ {'none','absolute','rope'}
    """
    def __init__(self, dim, num_heads=8, bias=True, proj_drop=0.,
                 pos_type="absolute", max_len=2048, use_layernorm=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.pos_type = pos_type
        self.max_len = max_len
        self.use_layernorm = use_layernorm

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)
        if use_layernorm:
            self.post_norm = nn.LayerNorm(dim)

        if pos_type == "absolute":
            self.abs_pe = nn.Parameter(torch.zeros(max_len, self.head_dim))
            nn.init.trunc_normal_(self.abs_pe, std=0.02)
        elif pos_type == "rope":
            theta = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
            theta = 1.0 / (10000 ** (theta / self.head_dim))
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            freqs = pos * theta.unsqueeze(0)
            cos = torch.zeros(max_len, self.head_dim)
            sin = torch.zeros(max_len, self.head_dim)
            cos[:, 0::2] = freqs.cos()
            sin[:, 0::2] = freqs.sin()
            cos[:, 1::2] = freqs.cos()
            sin[:, 1::2] = freqs.sin()
            self.register_buffer("rope_cos", cos.unsqueeze(0).unsqueeze(0))  # [1,1,L,Dh]
            self.register_buffer("rope_sin", sin.unsqueeze(0).unsqueeze(0))
        elif pos_type == "none":
            pass
        else:
            raise ValueError("pos_type must be 'none' | 'absolute' | 'rope'")

    def _apply_pos(self, q, k, N):
        if self.pos_type == "none":
            return q, k
        if self.pos_type == "absolute":
            pe = self.abs_pe[:N, :].unsqueeze(0).unsqueeze(0)  # [1,1,N,Dh]
            return q + pe, k + pe
        if self.pos_type == "rope":
            cos = self.rope_cos[:, :, :N, :]
            sin = self.rope_sin[:, :, :N, :]
            return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raise RuntimeError("Unknown pos_type")

    def forward(self, x):
        B, N, C = x.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Dh)
        q, k, v = qkv.unbind(dim=2)       # [B,N,H,Dh]
        q = q.transpose(1, 2)             # [B,H,N,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self._apply_pos(q, k, N)

        S = torch.matmul(k.transpose(-2, -1), v)  # [B,H,Dh,Dh]
        y = torch.matmul(q, S) / float(N)         # [B,H,N,Dh]

        y = y.transpose(1, 2).reshape(B, N, C)
        if self.use_layernorm:
            y = self.post_norm(y)
        y = self.drop(self.proj(y))
        return y


class QuadKernelAttention(nn.Module):
    """
    Quadratic kernel attention:
      X = (QK^T)/sqrt(d)
      use_psd_square=True :  K = gamma * (beta*X + alpha)^2          (PSD)
      use_psd_square=False:  K = (a2*c2)X^2 + (a1*c1)X + (a0*c0)
    """
    def __init__(self, dim, num_heads=8, bias=True, proj_drop=0.,
                 use_psd_square=True, c0=1.0, c1=1.0, c2=1.0, scale_by_len=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale_qk = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)
        self.scale_by_len = scale_by_len

        self.use_psd_square = use_psd_square
        if use_psd_square:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta  = nn.Parameter(torch.tensor(1.0))
            self.gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.a0 = nn.Parameter(torch.tensor(1.0))
            self.a1 = nn.Parameter(torch.tensor(1.0))
            self.a2 = nn.Parameter(torch.tensor(0.1))
            self.register_buffer("c0", torch.tensor(float(c0)))
            self.register_buffer("c1", torch.tensor(float(c1)))
            self.register_buffer("c2", torch.tensor(float(c2)))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        X = torch.matmul(q, k.transpose(-2, -1)) * self.scale_qk  # [B,H,N,N]

        if self.use_psd_square:
            K = self.gamma * (self.beta * X + self.alpha) ** 2
        else:
            K = self.a2 * self.c2 * (X * X) + self.a1 * self.c1 * X + self.a0 * self.c0

        if self.scale_by_len:
            K = K / float(N)

        y = torch.matmul(K, v)                # [B,H,N,Dh]
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.drop(self.proj(y))
        return y


class SelectableAttention(nn.Module):
    """
    Unified interface via attn_type:
      attn_type ∈ {'poly_kernel', 'pos_kernel', 'quad_kernel'}
    """
    def __init__(self, dim, num_heads=8, attn_type: str = "poly_kernel",
                 # common
                 bias=True, proj_drop=0.0,
                 # PolyKernelAttentionTimmCompat
                 degree=2, alpha=1.0, beta=1.0, average_by_len=True,
                 # LinearPositionalAttention
                 pos_type="absolute", max_len=2048, use_layernorm=True,
                 # QuadKernelAttention
                 use_psd_square=True, c0=1.0, c1=1.0, c2=1.0, scale_by_len=True):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "poly_kernel":
            self.impl = PolyKernelAttentionTimmCompat(
                dim=dim, num_heads=num_heads, qkv_bias=bias,
                attn_drop=proj_drop, proj_drop=proj_drop,
                degree=degree, alpha=alpha, beta=beta, average_by_len=average_by_len
            )
        elif attn_type == "pos_kernel":
            self.impl = LinearPositionalAttention(
                dim=dim, num_heads=num_heads, bias=bias, proj_drop=proj_drop,
                pos_type=pos_type, max_len=max_len, use_layernorm=use_layernorm
            )
        elif attn_type == "quad_kernel":
            self.impl = QuadKernelAttention(
                dim=dim, num_heads=num_heads, bias=bias, proj_drop=proj_drop,
                use_psd_square=use_psd_square, c0=c0, c1=c1, c2=c2, scale_by_len=scale_by_len
            )
        else:
            raise ValueError("attn_type must be 'poly_kernel' | 'pos_kernel' | 'quad_kernel'")

    def forward(self, x):
        return self.impl(x)


# =========================
# ViT Replacement Helpers
# =========================
def _extract_timm_attn_hparams(old_attn: nn.Module):
    """Best-effort extraction of timm Attention hyper-params."""
    dim = old_attn.qkv.in_features
    num_heads = getattr(old_attn, "num_heads", None)
    attn_drop = float(old_attn.attn_drop.p) if isinstance(getattr(old_attn, "attn_drop", None), nn.Dropout) else 0.0
    proj_drop = float(old_attn.proj_drop.p) if isinstance(getattr(old_attn, "proj_drop", None), nn.Dropout) else 0.0
    return dim, num_heads, attn_drop, proj_drop


def replace_vit_with_poly_attention(model: nn.Module,
                                    degree=2, alpha=1.0, beta=1.0,
                                    average_by_len=True):
    """
    Backward-compatible helper: replace each block.attn with PolyKernelAttentionTimmCompat.
    """
    for blk in model.blocks:
        old_attn = blk.attn
        dim, num_heads, attn_drop, proj_drop = _extract_timm_attn_hparams(old_attn)
        new_attn = PolyKernelAttentionTimmCompat(
            dim=dim, num_heads=num_heads, qkv_bias=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
            degree=degree, alpha=alpha, beta=beta, average_by_len=average_by_len
        )
        nn.init.trunc_normal_(new_attn.qkv.weight, std=0.02)
        if new_attn.qkv.bias is not None:
            nn.init.zeros_(new_attn.qkv.bias)
        nn.init.trunc_normal_(new_attn.proj.weight, std=0.02)
        if new_attn.proj.bias is not None:
            nn.init.zeros_(new_attn.proj.bias)
        blk.attn = new_attn
    return model


def replace_vit_attention(model: nn.Module,
                          attn_type: str = "poly_kernel",
                          # common
                          bias=True, proj_drop=0.0,
                          # PolyKernelAttentionTimmCompat
                          degree=2, alpha=1.0, beta=1.0, average_by_len=True,
                          # LinearPositionalAttention
                          pos_type="absolute", max_len=2048, use_layernorm=True,
                          # QuadKernelAttention
                          use_psd_square=True, c0=1.0, c1=1.0, c2=1.0, scale_by_len=True):
    """
    Generic replacer to swap timm ViT block.attn with a selectable attention.
      attn_type ∈ {'poly_kernel', 'pos_kernel', 'quad_kernel'}
    """
    for blk in model.blocks:
        old_attn = blk.attn
        dim, num_heads, attn_drop_old, proj_drop_old = _extract_timm_attn_hparams(old_attn)

        # prefer using the model's original drop rates unless explicitly overridden by proj_drop
        final_proj_drop = proj_drop if proj_drop is not None else proj_drop_old
        final_attn_drop = attn_drop_old  # used by poly kernel path (for K drop)

        if attn_type == "poly_kernel":
            new_attn = PolyKernelAttentionTimmCompat(
                dim=dim, num_heads=num_heads, qkv_bias=bias,
                attn_drop=final_attn_drop, proj_drop=final_proj_drop,
                degree=degree, alpha=alpha, beta=beta, average_by_len=average_by_len
            )
        elif attn_type == "pos_kernel":
            new_attn = LinearPositionalAttention(
                dim=dim, num_heads=num_heads, bias=bias, proj_drop=final_proj_drop,
                pos_type=pos_type, max_len=max_len, use_layernorm=use_layernorm
            )
        elif attn_type == "quad_kernel":
            new_attn = QuadKernelAttention(
                dim=dim, num_heads=num_heads, bias=bias, proj_drop=final_proj_drop,
                use_psd_square=use_psd_square, c0=c0, c1=c1, c2=c2, scale_by_len=scale_by_len
            )
        else:
            raise ValueError("attn_type must be 'poly_kernel' | 'pos_kernel' | 'quad_kernel'")

        # optional init similar to timm
        if hasattr(new_attn, "qkv"):
            nn.init.trunc_normal_(new_attn.qkv.weight, std=0.02)
            if new_attn.qkv.bias is not None:
                nn.init.zeros_(new_attn.qkv.bias)
        if hasattr(new_attn, "proj"):
            nn.init.trunc_normal_(new_attn.proj.weight, std=0.02)
            if new_attn.proj.bias is not None:
                nn.init.zeros_(new_attn.proj.bias)

        blk.attn = new_attn

    return model


class PolyNorm(nn.Module):
    def __init__(self, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.is_setup = False
        self.number = 0
        self.use_running_var_mean = False

        self.k = 4
        self.mu = 2
        
        self.register_buffer('num_batches_tracked', torch.zeros(1))

        self.normalized_shape = None
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.saved_var = None

        self.gain = None
        self.bias = None

    def setup(self, custom_settings):
        self.is_setup = True

        self.k = custom_settings.k
        self.mu = custom_settings.mu

        self.filter_var_mean = self.k * self.mu
        self.g_2 = 1 / (4 * (self.k - 1) * self.mu**0.5)
        self.g_3 = (5 - self.k) / (4 * self.mu**0.5)

    def forward(self, x: torch.Tensor):
        assert self.is_setup, "MyLayerNorm needs to be explicitly setup before forward pass."

        if self.normalized_shape is None:
            self.normalized_shape = x.size()[1:]

        exponential_average_factor = 0.0
        if self.training:
            self.num_batches_tracked += 1
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        
        dims = tuple(range(1, x.ndim))
        
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)

        var = mean_x2 - mean ** 2  

        var_mean = var.mean().squeeze()

        if not hasattr(self, 'running_var_mean'):
            self.register_buffer('running_var_mean', torch.ones_like(var_mean))
            if self.elementwise_affine:
                self.gain = nn.Parameter(torch.ones(self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

        self.saved_var_mean = var_mean

        # if self.training and self.filter_var_mean > 0:
        #     if (var_mean > self.running_var_mean * self.filter_var_mean).any():
        #         self.filter_var_mean_times = 1
        #         x_norm = (x - mean) / torch.sqrt(var + self.eps)
        #         if self.elementwise_affine:
        #             x_norm = x_norm * self.gain + self.bias
        #         return x_norm
        #     else:
        #         self.filter_var_mean_times = 0

        _running_var_mean = self.running_var_mean

        if False:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            if self.training and not self.use_running_var_mean:
                final_var_mean = var_mean
            else:
                final_var_mean = _running_var_mean

            v = var / final_var_mean   

            f_result = torch.sqrt(1 / (self.mu * v))
            g_result = (v - self.k) ** 2 * self.g_2 + self.g_3

            if self.training :
                g_result[v > self.k] = f_result[v > self.k]
                # g_result[v > self.k] = self.g_3
                ratio = (v > self.k).float().mean()
                self.filter_var_mean_times = ratio.item()

            x_norm = (x - mean) * g_result * torch.sqrt(self.mu / final_var_mean)

        with torch.no_grad():
            if self.training:
                self.running_var_mean = (
                    (1 - exponential_average_factor) * self.running_var_mean
                    + exponential_average_factor * self.saved_var_mean
                )

        if self.elementwise_affine:
            x_norm = x_norm * self.gain + self.bias

        return x_norm
    

class poly_act(nn.Module):
    """
    Polynomial activation replacement for ViT.
    Evaluates y = sum_{i=0..degree} (w_i * c_i) * x^i along the last dim (hidden_dim).
    - weights per feature dim (D), shared across batch/sequence.
    - supports inputs of shape [..., D] (e.g., [B, T, D] or [B, D]).
    """
    def __init__(self, weight_inits, factors, act_degree, act_dropout):
        super().__init__()
        self.num_features = None              # D (hidden_dim)
        self.weight_inits = weight_inits      # list length = degree+1
        self.factors = nn.Parameter(torch.as_tensor(factors, dtype=torch.float32),
                                    requires_grad=False)  # length = degree+1
        self.act_degree = act_degree
        self.dropout = nn.Dropout(act_dropout)

        if len(weight_inits) != self.act_degree + 1:
            raise ValueError("weight_inits must be of length act_degree + 1")
        if len(factors) != self.act_degree + 1:
            raise ValueError("factors must be of length act_degree + 1")

        self.weight = None  # will be initialized on first forward: shape [D, degree+1]

    def forward(self, x: torch.Tensor):
        # x shape: [..., D]
        if self.num_features is None:
            self.num_features = x.size(-1)

            # Initialize per-feature coefficients: [D, degree+1]
            # Column i is initialized to weight_inits[i] for all features.
            device, dtype = x.device, x.dtype
            init_cols = [
                torch.full((self.num_features,), w_init, device=device, dtype=dtype)
                for w_init in self.weight_inits
            ]
            initial_weights = torch.stack(init_cols, dim=-1)  # [D, degree+1]
            self.weight = nn.Parameter(initial_weights, requires_grad=True)

        # Cast factors to x's dtype/device for safe math
        f = self.factors.to(dtype=x.dtype, device=x.device)       # [degree+1]
        w = self.weight.to(dtype=x.dtype, device=x.device)        # [D, degree+1]

        # Horner scheme along last dim (per feature coefficients)
        # Start with constant term: w[:,0] * f[0] -> shape [D], broadcast to [..., D]
        y = w[:, 0] * f[0]                                        # [D]
        for d in range(self.act_degree):
            # y * x -> [..., D]; add next coefficient term (w[:, d+1] * f[d+1]) -> [D] (broadcast)
            y = y * x + (w[:, d + 1] * f[d + 1])

        return self.dropout(y)