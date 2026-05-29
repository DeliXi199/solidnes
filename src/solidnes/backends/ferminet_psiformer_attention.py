"""SolidNES attention implementations for FermiNet PsiFormer configs.

The upstream PsiFormer implementation projects Q, K, and V with three separate
matmuls.  For self-attention, LapNet uses the more conventional fused QKV
projection.  This module installs that fused projection as an optional
process-local patch while keeping the public FermiNet network API unchanged.
"""

from __future__ import annotations

import os
from typing import Any


PSIFORMER_KWARG_KEYS = (
    "num_layers",
    "num_heads",
    "heads_dim",
    "mlp_hidden_dims",
    "use_layer_norm",
    "tf32",
)

_ATTENTION_KERNEL_GPU = "jax"


def psiformer_kwargs_from_config(psiformer_cfg: Any) -> dict[str, Any]:
    """Return only upstream PsiFormer kwargs accepted by FermiNet."""

    return {key: psiformer_cfg[key] for key in PSIFORMER_KWARG_KEYS}


def psiformer_attention_implementation(cfg: Any) -> str | None:
    """Return the requested PsiFormer attention implementation."""

    if cfg.network.network_type != "psiformer":
        return None
    return str(cfg.network.get("psiformer_attention_implementation", "auto"))


def resolved_psiformer_attention_kernel_gpu() -> str:
    """Return the process-local GPU attention kernel selected at install time."""

    return _ATTENTION_KERNEL_GPU


def install_psiformer_attention_implementation(cfg: Any) -> str | None:
    """Install the configured PsiFormer attention implementation.

    FermiNet constructs PsiFormer attention blocks by calling
    ``ferminet.psiformer.make_multi_head_attention`` at network construction
    time.  Replacing that factory before construction is enough for both
    SolidNES adapter builds and the native FermiNet training entry point.
    """

    implementation = psiformer_attention_implementation(cfg)
    if implementation is None:
        return None

    import jax  # pylint: disable=import-outside-toplevel
    from ferminet import psiformer  # pylint: disable=import-outside-toplevel

    if jax.default_backend() == "cpu" and bool(cfg.network.psiformer.get("tf32", False)):
        cfg.network.psiformer.tf32 = False

    original_name = "_solidnes_original_make_multi_head_attention"
    if not hasattr(psiformer, original_name):
        setattr(psiformer, original_name, psiformer.make_multi_head_attention)
    original = getattr(psiformer, original_name)

    if implementation == "auto":
        implementation = "fused_qkv"
    _set_attention_kernel_options(cfg)

    if implementation in {"ferminet", "upstream", "default"}:
        psiformer.make_multi_head_attention = make_folx_aware_multi_head_attention
    elif implementation == "fused_qkv":
        psiformer.make_multi_head_attention = make_fused_qkv_multi_head_attention
    else:
        raise ValueError(
            "Unsupported PsiFormer attention implementation: "
            f"{implementation!r}. Expected 'auto', 'ferminet', or 'fused_qkv'."
        )
    _install_folx_stable_psiformer_layers(psiformer)
    return implementation


def _set_attention_kernel_options(cfg: Any) -> None:
    """Apply process-local attention kernel options from config/env."""

    global _ATTENTION_KERNEL_GPU
    configured = cfg.network.get("psiformer_attention_kernel_gpu", None)
    env_value = os.environ.get("SOLIDNES_PSIFORMER_ATTENTION_KERNEL_GPU")
    kernel = str(env_value or configured or _ATTENTION_KERNEL_GPU)
    if kernel not in {"pallas", "reference", "jax"}:
        raise ValueError(
            "Unsupported PsiFormer GPU attention kernel: "
            f"{kernel!r}. Expected 'pallas', 'reference', or 'jax'."
        )
    _ATTENTION_KERNEL_GPU = kernel


def _install_folx_stable_psiformer_layers(psiformer: Any) -> None:
    """Patch PsiFormer input concatenation to avoid a FOLX x64 mask fallback."""

    original_name = "_solidnes_original_make_psiformer_layers"
    patched_name = "_solidnes_folx_stable_make_psiformer_layers"
    if not hasattr(psiformer, original_name):
        setattr(psiformer, original_name, psiformer.make_psiformer_layers)
    if not hasattr(psiformer, patched_name):
        setattr(
            psiformer,
            patched_name,
            _make_folx_stable_psiformer_layers_factory(psiformer),
        )
    psiformer.make_psiformer_layers = getattr(psiformer, patched_name)


def _make_folx_stable_psiformer_layers_factory(psiformer: Any):
    """Return a FermiNet-compatible ``make_psiformer_layers`` replacement."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from ferminet import network_blocks  # pylint: disable=import-outside-toplevel

    def make_psiformer_layers(nspins, natoms, options):
        del nspins, natoms

        attn_dim = options.num_heads * options.heads_dim
        self_attn_init, self_attn_apply = psiformer.make_self_attention_block(
            num_layers=options.num_layers,
            num_heads=options.num_heads,
            heads_dim=options.heads_dim,
            mlp_hidden_dims=options.mlp_hidden_dims,
            use_layer_norm=options.use_layer_norm,
            tf32=options.tf32,
        )

        def init(key):
            params = {}
            key, _ = jax.random.split(key)
            feature_dims, params["input"] = options.feature_layer.init()
            one_electron_feature_dim, _ = feature_dims
            feature_dim = one_electron_feature_dim + 1

            key, subkey = jax.random.split(key)
            params["embed"] = network_blocks.init_linear_layer(
                subkey, in_dim=feature_dim, out_dim=attn_dim, include_bias=False
            )["w"]

            key, subkey = jax.random.split(key)
            params.update(self_attn_init(subkey, attn_dim))
            return attn_dim, params

        def apply(
            params,
            *,
            ae,
            r_ae,
            ee,
            r_ee,
            spins,
            charges,
        ):
            del charges
            ae_features, _ = options.feature_layer.apply(
                ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params["input"]
            )
            spin_features = jnp.asarray(spins[..., None], dtype=ae_features.dtype)
            spin_features = spin_features + ae_features[..., :1] * 0.0
            features = jnp.concatenate((ae_features, spin_features), axis=-1)
            x = jnp.dot(features, params["embed"])
            return self_attn_apply(params, x)

        return init, apply

    return make_psiformer_layers


def make_folx_aware_multi_head_attention(
    num_heads: int,
    heads_dim: int,
    tf32: bool = False,
):
    """FermiNet-shaped Q/K/V projections with a FOLX-aware attention core."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from ferminet import network_blocks  # pylint: disable=import-outside-toplevel

    prec = jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if tf32 else None

    def linear_projection(x, weights):
        y = jnp.dot(x, weights, precision=prec)
        return y.reshape(*x.shape[:-1], num_heads, heads_dim)

    def init(key, q_d: int, kv_d: int, output_channels: int | None = None):
        qkv_hiddens = num_heads * heads_dim
        if not output_channels:
            output_channels = qkv_hiddens

        key, *subkeys = jax.random.split(key, num=4)
        params = {
            "q_w": network_blocks.init_linear_layer(
                subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False
            )["w"],
            "k_w": network_blocks.init_linear_layer(
                subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
            )["w"],
            "v_w": network_blocks.init_linear_layer(
                subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
            )["w"],
        }

        key, subkey = jax.random.split(key)
        params["attn_output"] = network_blocks.init_linear_layer(
            subkey,
            in_dim=qkv_hiddens,
            out_dim=output_channels,
            include_bias=False,
        )["w"]
        return params

    def apply(params, query, key, value):
        q = linear_projection(query, params["q_w"])
        k = linear_projection(key, params["k_w"])
        v = linear_projection(value, params["v_w"])
        attn = _self_attention(q, k, v, heads_dim=heads_dim)
        attn = jnp.reshape(attn, (*query.shape[:-1], num_heads * heads_dim))
        return network_blocks.linear_layer(attn, params["attn_output"])

    return init, apply


def make_fused_qkv_multi_head_attention(
    num_heads: int,
    heads_dim: int,
    tf32: bool = False,
):
    """FermiNet-compatible multi-head self-attention with fused QKV matmul."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from ferminet import network_blocks  # pylint: disable=import-outside-toplevel

    prec = jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if tf32 else None

    def init(
        key,
        q_d: int,
        kv_d: int,
        output_channels: int | None = None,
    ):
        if q_d != kv_d:
            raise ValueError(
                "fused_qkv PsiFormer attention only supports self-attention "
                f"with q_d == kv_d; got {q_d} and {kv_d}."
            )

        qkv_hiddens = num_heads * heads_dim
        if not output_channels:
            output_channels = qkv_hiddens

        key, *subkeys = jax.random.split(key, num=4)
        q_w = network_blocks.init_linear_layer(
            subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        k_w = network_blocks.init_linear_layer(
            subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        v_w = network_blocks.init_linear_layer(
            subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]

        key, subkey = jax.random.split(key)
        return {
            "qkv_w": jnp.concatenate((q_w, k_w, v_w), axis=-1),
            "attn_output": network_blocks.init_linear_layer(
                subkey,
                in_dim=qkv_hiddens,
                out_dim=output_channels,
                include_bias=False,
            )["w"],
        }

    def apply(params, query, key, value):
        if query is not key or query is not value:
            raise ValueError("fused_qkv PsiFormer attention only supports self-attention.")

        qkv_hiddens = num_heads * heads_dim
        qkv = jnp.dot(query, params["qkv_w"], precision=prec)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(*query.shape[:-1], num_heads, heads_dim)
        k = k.reshape(*query.shape[:-1], num_heads, heads_dim)
        v = v.reshape(*query.shape[:-1], num_heads, heads_dim)
        attn = _self_attention(q, k, v, heads_dim=heads_dim)
        attn = jnp.reshape(attn, (*query.shape[:-1], qkv_hiddens))
        return network_blocks.linear_layer(attn, params["attn_output"])

    return init, apply


def _self_attention(q, k, v, *, heads_dim: int):
    """Compute scaled self-attention through FOLX's registered attention rule."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    kernel = _attention_kernel(jax.default_backend())
    if kernel == "jax":
        return _jax_self_attention(q, k, v, heads_dim=heads_dim, jax=jax, jnp=jnp, np=np)

    try:
        from folx.experimental.pallas.attention import (  # pylint: disable=import-outside-toplevel
            multi_head_self_attention,
        )
    except Exception:  # pragma: no cover - fallback for environments without pallas.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", q, k)
        attn_logits *= 1.0 / np.sqrt(heads_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        return jnp.einsum("...htT,...Thd->...thd", attn_weights, v)

    original_shape = q.shape
    seq_len = int(original_shape[-3])
    leading_shape = tuple(original_shape[:-3])
    batch_size = int(np.prod(leading_shape, dtype=int)) if leading_shape else 1

    q = jnp.reshape(q, (batch_size, seq_len, original_shape[-2], original_shape[-1]))
    k = jnp.reshape(k, (batch_size, seq_len, original_shape[-2], original_shape[-1]))
    v = jnp.reshape(v, (batch_size, seq_len, original_shape[-2], original_shape[-1]))

    padded_seq_len = _attention_padded_sequence_length(seq_len)
    if padded_seq_len != seq_len:
        pad_len = padded_seq_len - seq_len
        q = _pad_attention_sequence(q, pad_len, jnp)
        k = _pad_attention_sequence(k, pad_len, jnp)
        v = _pad_attention_sequence(v, pad_len, jnp)

    scale = 1.0 / np.sqrt(heads_dim)
    mask = jnp.arange(padded_seq_len) < seq_len
    mask = jnp.broadcast_to(mask[None, :], (batch_size, padded_seq_len))
    input_mask = jnp.ones((seq_len * 3, batch_size), dtype=bool)
    attn = multi_head_self_attention(
        q * scale,
        k,
        v,
        mask,
        input_mask,
        None,
        kernel=kernel,
        q_block_len=None,
    )
    attn = attn[:, :seq_len]
    return jnp.reshape(attn, original_shape)


def _attention_padded_sequence_length(seq_len: int) -> int:
    """Return an attention sequence length accepted by the FOLX attention wrapper."""

    return max(16, 1 << (seq_len - 1).bit_length())


def _attention_kernel(backend: str) -> str:
    """Return the FOLX attention kernel for the active JAX backend."""

    return "reference" if backend == "cpu" else _ATTENTION_KERNEL_GPU


def _jax_self_attention(q, k, v, *, heads_dim: int, jax, jnp, np):
    attn_logits = jnp.einsum("...thd,...Thd->...htT", q, k)
    attn_logits *= 1.0 / np.sqrt(heads_dim)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    return jnp.einsum("...htT,...Thd->...thd", attn_weights, v)


def _pad_attention_sequence(x, pad_len: int, jnp):
    pad = jnp.broadcast_to(x[:, :1] * 0.0, (x.shape[0], pad_len, x.shape[2], x.shape[3]))
    return jnp.concatenate((x, pad), axis=1)
