"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.

MLX implementation of the Cellpose-SAM (CP-SAM) Transformer model for Apple Silicon.
This module re-implements the ViT-based encoder and cellpose readout head using MLX,
enabling native Apple Silicon GPU acceleration without PyTorch/MPS overhead.
"""

import math
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create a dummy base class so the module can be imported without MLX
    class _DummyModule:
        pass


def check_mlx():
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX is not installed. Install it with: pip install mlx\n"
            "MLX requires macOS 13.5+ on Apple Silicon (M1/M2/M3/M4)."
        )


_BaseModule = nn.Module if MLX_AVAILABLE else _DummyModule


class MLPBlock(_BaseModule):
    """MLP block used in each transformer layer."""

    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def __call__(self, x):
        return self.lin2(nn.gelu(self.lin1(x)))


class LayerNorm2d(_BaseModule):
    """Layer normalization for 2D feature maps in NHWC layout."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x):
        # x: (N, H, W, C) — NHWC layout, normalize over C
        u = mx.mean(x, axis=-1, keepdims=True)
        s = mx.mean((x - u) ** 2, axis=-1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class Attention(_BaseModule):
    """Multi-head attention block with relative position embeddings."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 use_rel_pos: bool = True, input_size: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = mx.zeros((2 * input_size - 1, self.head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size - 1, self.head_dim))
            # Precomputed index tables (set by _precompute_rel_pos)
            self._rel_h_idx = None
            self._rel_w_idx = None

    def __call__(self, x):
        B, H, W, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        # QKV projection: (B, H, W, 3*dim) -> (B, H*W, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, num_heads, head_dim)
        q = qkv[:, :, 0]  # (B, H*W, num_heads, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if self.use_rel_pos:
            # Transpose to (B, num_heads, H*W, head_dim) for attention
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Scaled dot-product attention
            attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # (B, heads, HW, HW)

            # Add relative position bias
            attn = self._add_rel_pos_bias(attn, q, H, W)

            attn = mx.softmax(attn, axis=-1)
            x = attn @ v  # (B, num_heads, H*W, head_dim)
            x = x.transpose(0, 2, 1, 3).reshape(B, H, W, -1)
        else:
            # Use fused SDPA when no rel_pos needed (no bias term)
            # q/k/v: (B, H*W, num_heads, head_dim)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            x = x.transpose(0, 2, 1, 3).reshape(B, H, W, -1)

        x = self.proj(x)
        return x

    def _add_rel_pos_bias(self, attn, q, H, W):
        """Add decomposed relative position bias to attention scores.

        Uses precomputed index tables and einsum for efficiency.
        """
        B_heads = attn.shape[0] * attn.shape[1]
        head_dim = self.head_dim

        # Gather rel_pos embeddings using precomputed indices
        Rh = self.rel_pos_h[self._rel_h_idx]  # (H, H, head_dim)
        Rw = self.rel_pos_w[self._rel_w_idx]  # (W, W, head_dim)

        # Reshape q for spatial decomposition: (B, heads, H, W, head_dim)
        r_q = q.reshape(q.shape[0], q.shape[1], H, W, head_dim)

        # rel_h: einsum "bnhwc,hkc->bnhwk" -> (B, heads, H, W, H)
        rel_h = mx.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
        # rel_w: einsum "bnhwc,wkc->bnhwk" -> (B, heads, H, W, W)
        rel_w = mx.einsum("bnhwc,wkc->bnhwk", r_q, Rw)

        # Add to attention: reshape attn to (B, heads, H, W, H, W)
        attn = (attn.reshape(q.shape[0], q.shape[1], H, W, H, W)
                + mx.expand_dims(rel_h, axis=-1)
                + mx.expand_dims(rel_w, axis=-2))
        attn = attn.reshape(q.shape[0], q.shape[1], H * W, H * W)
        return attn


def _precompute_rel_pos_for_size(spatial_size, rel_pos):
    """Interpolate relative position embeddings to match the target spatial size."""
    target_len = 2 * spatial_size - 1
    if rel_pos.shape[0] == target_len:
        return rel_pos
    from scipy.interpolate import interp1d
    rel_pos_np = np.array(rel_pos)
    x_old = np.linspace(0, 1, rel_pos_np.shape[0])
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, rel_pos_np, axis=0, kind='linear')
    return mx.array(f(x_new).astype(np.float32))


def _precompute_rel_pos_indices(spatial_size):
    """Precompute relative position index table for a given spatial size.

    Returns an int32 mx.array of shape (spatial_size, spatial_size) that
    can be used to gather from the rel_pos embedding.
    """
    coords = np.arange(spatial_size)
    # relative_coords[i,j] = i - j + (spatial_size - 1)
    relative_coords = coords[:, None] - coords[None, :] + (spatial_size - 1)
    return mx.array(relative_coords.astype(np.int32))


class Block(_BaseModule):
    """Transformer block with global attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, norm_eps: float = 1e-6,
                 use_rel_pos: bool = True, input_size: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              use_rel_pos=use_rel_pos, input_size=input_size)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = 0  # always global attention in cellpose

    def __call__(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(_BaseModule):
    """Image to Patch Embedding using convolution."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 1024, ps: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=ps, stride=ps, bias=True)

    def __call__(self, x):
        # x: (N, C, H, W) in channels-first format
        # MLX Conv2d expects (N, H, W, C), so transpose
        x = x.transpose(0, 2, 3, 1)  # (N, H, W, C)
        x = self.proj(x)  # (N, H', W', embed_dim) - already channels-last
        return x


class Neck(_BaseModule):
    """Neck module: two Conv2d + LayerNorm2d layers, all in NHWC layout."""

    def __init__(self, embed_dim: int = 1024, out_chans: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False)
        self.ln1 = LayerNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.ln2 = LayerNorm2d(out_chans)

    def __call__(self, x):
        # x: (N, H, W, C) — already NHWC from transformer blocks
        x = self.conv1(x)   # (N, H, W, out_chans)
        x = self.ln1(x)     # NHWC LayerNorm over C
        x = self.conv2(x)   # (N, H, W, out_chans)
        x = self.ln2(x)     # NHWC LayerNorm over C
        return x


class TransformerMLX(_BaseModule):
    """
    MLX implementation of the Cellpose-SAM Transformer model.

    This mirrors the PyTorch Transformer class in vit_sam.py but uses MLX operations.
    Only used for inference (not training).
    """

    def __init__(self, backbone="vit_l", ps=8, nout=3, bsize=256):
        super().__init__()

        # ViT-Large config (default for cellpose)
        configs = {
            "vit_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "vit_h": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
            "vit_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        }
        config = configs[backbone]
        embed_dim = config["embed_dim"]
        depth = config["depth"]
        num_heads = config["num_heads"]

        self.ps = ps
        self.nout = nout
        self.bsize = bsize

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dim, ps=ps)

        # Positional embedding
        pos_embed_size = bsize // ps
        self.pos_embed = mx.zeros((1, pos_embed_size, pos_embed_size, embed_dim))

        # Transformer blocks (input_size = pos_embed_size for rel_pos)
        self.blocks = [
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                  qkv_bias=True, norm_eps=1e-6, use_rel_pos=True,
                  input_size=pos_embed_size)
            for _ in range(depth)
        ]

        # Neck
        self.neck = Neck(embed_dim=embed_dim, out_chans=256)

        # Readout head
        self.out_conv = nn.Conv2d(256, nout * ps ** 2, kernel_size=1, bias=True)

        # W2: fixed reshape matrix (not trainable) for transpose convolution equivalent
        self.W2 = mx.array(
            np.eye(nout * ps ** 2).reshape(nout * ps ** 2, nout, ps, ps).astype(np.float32)
        )

        # Diameter parameters (loaded from checkpoint)
        self.diam_labels = mx.array([30.0])
        self.diam_mean = mx.array([30.0])

        # Compiled forward function (set after weight loading)
        self._compiled_forward = None

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (N, C, H, W) in channels-first format.

        Returns:
            Tuple of (output, style):
                output: (N, nout, H, W) flow predictions
                style: (N, 256) zeros for backward compatibility
        """
        if self._compiled_forward is not None:
            return self._compiled_forward(x)
        return self._forward(x)

    def _forward(self, x):
        """Uncompiled forward pass."""
        # Patch embedding: (N, C, H, W) -> (N, H', W', embed_dim)
        x = self.patch_embed(x)

        # Add positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Transformer blocks — x stays in (N, H', W', embed_dim) throughout
        for blk in self.blocks:
            x = blk(x)

        # Neck: (N, H', W', embed_dim) -> (N, H', W', 256), all NHWC
        x = self.neck(x)

        # Readout: (N, H', W', 256) -> (N, H', W', nout*ps^2)
        x1 = self.out_conv(x)
        # Convert to NCHW for pixel shuffle
        x1 = x1.transpose(0, 3, 1, 2)  # (N, nout*ps^2, H', W')

        # Transpose convolution equivalent (pixel shuffle)
        x1 = self._conv_transpose(x1)

        # Style vector (zeros for compatibility)
        style = mx.zeros((x.shape[0], 256))

        return x1, style

    def _conv_transpose(self, x):
        """
        Equivalent to F.conv_transpose2d(x, W2, stride=ps, padding=0).
        Implemented as pixel shuffle since W2 is an identity reshape matrix.

        Args:
            x: (N, nout*ps^2, H', W')

        Returns:
            (N, nout, H'*ps, W'*ps)
        """
        N, C, H, W = x.shape
        ps = self.ps
        nout = self.nout
        # Reshape: (N, nout, ps, ps, H, W)
        x = x.reshape(N, nout, ps, ps, H, W)
        # Permute to (N, nout, H, ps, W, ps)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        # Reshape to final: (N, nout, H*ps, W*ps)
        x = x.reshape(N, nout, H * ps, W * ps)
        return x

    def load_weights_from_pytorch(self, state_dict):
        """
        Load weights from a PyTorch state dict, converting to MLX format.

        Args:
            state_dict: Dictionary of PyTorch weight tensors (as numpy arrays).
        """
        from .mlx_utils import convert_pytorch_to_mlx_weights
        mlx_weights = convert_pytorch_to_mlx_weights(state_dict, self)
        self.update(mlx_weights)
        mx.eval(self.parameters())
        self._precompute_rel_pos()

        # Compile the forward pass for faster inference
        self._compiled_forward = mx.compile(self._forward)

    def _precompute_rel_pos(self):
        """Pre-compute interpolated relative position embeddings and index tables.

        The SAM checkpoint stores rel_pos at size (2*64-1, head_dim)=(127, head_dim),
        but cellpose uses spatial size bsize//ps=32, needing (2*32-1)=(63, head_dim).
        Pre-computing avoids repeated scipy interpolation and numpy ops during inference.
        """
        spatial_size = self.bsize // self.ps
        # Precompute index tables once
        h_idx = _precompute_rel_pos_indices(spatial_size)
        w_idx = _precompute_rel_pos_indices(spatial_size)
        for blk in self.blocks:
            if blk.attn.use_rel_pos:
                blk.attn.rel_pos_h = _precompute_rel_pos_for_size(
                    spatial_size, blk.attn.rel_pos_h)
                blk.attn.rel_pos_w = _precompute_rel_pos_for_size(
                    spatial_size, blk.attn.rel_pos_w)
                blk.attn._rel_h_idx = h_idx
                blk.attn._rel_w_idx = w_idx
        mx.eval(self.parameters())
