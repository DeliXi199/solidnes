#!/usr/bin/env python
"""Benchmark upstream and fused-QKV PsiFormer attention forward passes."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    "configs/experiment/diamond_c_psiformer_pbc_gamma_attention_build_benchmark.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["ferminet", "fused_qkv"],
        choices=["ferminet", "fused_qkv"],
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to the experiment validation_dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    platform = _normalize_platform(args.platform)
    if platform:
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)

    from solidnes.backends.ferminet_adapter import build_ferminet_adapter
    from solidnes.excited_states.ferminet_pbc_adapter import (
        build_external_state_adapter,
        configure_jax_platform,
    )

    configure_jax_platform(platform)
    bundle = build_ferminet_adapter(args.experiment)
    output_dir = (
        PROJECT_ROOT / bundle.experiment["output"]["validation_dir"]
        if args.output_dir is None
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    baseline = None
    for implementation in args.implementations:
        cfg = copy.deepcopy(bundle.cfg)
        cfg.system.states = 0
        cfg.network.psiformer_attention_implementation = implementation
        if platform == "cpu":
            cfg.network.psiformer.tf32 = False
        adapter = build_external_state_adapter(cfg, platform=platform)
        jax = adapter.modules.jax
        jnp = adapter.modules.jnp

        key = jax.random.PRNGKey(args.seed)
        key, params_key, samples_key = jax.random.split(key, 3)
        state_params = adapter.init_state_params(params_key, 2)
        samples = adapter.tiny_state_samples(
            samples_key,
            nstates=2,
            walkers=args.walkers,
        )
        positions, spins, atoms, charges = samples.for_sample_state(0)
        params = state_params[0]

        @jax.jit
        def forward(p, x, s, a, z):
            sign, logabs = adapter.batched_signed_network(p, x, s, a, z)
            return sign, logabs

        for _ in range(args.warmup):
            sign, logabs = forward(params, positions, spins, atoms, charges)
            jax.block_until_ready((sign, logabs))

        times = []
        last_sign = None
        last_logabs = None
        for _ in range(args.repeats):
            start = perf_counter()
            sign, logabs = forward(params, positions, spins, atoms, charges)
            jax.block_until_ready((sign, logabs))
            times.append(perf_counter() - start)
            last_sign = sign
            last_logabs = logabs

        assert last_sign is not None and last_logabs is not None
        record: dict[str, Any] = {
            "implementation": implementation,
            "platform": jax.default_backend(),
            "walkers": args.walkers,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "mean_seconds": float(jnp.asarray(times).mean()),
            "median_seconds": float(jnp.median(jnp.asarray(times))),
            "min_seconds": float(jnp.asarray(times).min()),
            "max_seconds": float(jnp.asarray(times).max()),
            "sign_shape": list(last_sign.shape),
            "logabs_shape": list(last_logabs.shape),
            "all_finite": bool(jnp.all(jnp.isfinite(last_logabs))),
        }
        if baseline is None:
            baseline = (last_sign, last_logabs, record)
            record["max_abs_logabs_delta_vs_first"] = 0.0
            record["max_abs_sign_delta_vs_first"] = 0.0
            record["speedup_vs_first_mean"] = 1.0
            record["speedup_vs_first_median"] = 1.0
        else:
            base_sign, base_logabs, base_record = baseline
            record["max_abs_logabs_delta_vs_first"] = float(
                jnp.max(jnp.abs(last_logabs - base_logabs))
            )
            record["max_abs_sign_delta_vs_first"] = float(
                jnp.max(jnp.abs(last_sign - base_sign))
            )
            record["speedup_vs_first_mean"] = (
                base_record["mean_seconds"] / record["mean_seconds"]
            )
            record["speedup_vs_first_median"] = (
                base_record["median_seconds"] / record["median_seconds"]
            )
        results.append(record)

    summary = {
        "experiment": str(bundle.paths.experiment.relative_to(PROJECT_ROOT)),
        "network_type": bundle.cfg.network.network_type,
        "psiformer_attention_implementation": (
            bundle.summary.psiformer_attention_implementation
        ),
        "num_layers": int(bundle.cfg.network.psiformer.num_layers),
        "num_heads": int(bundle.cfg.network.psiformer.num_heads),
        "heads_dim": int(bundle.cfg.network.psiformer.heads_dim),
        "mlp_hidden_dims": list(bundle.cfg.network.psiformer.mlp_hidden_dims),
        "results": results,
    }
    tag = f"{results[0]['platform']}_w{args.walkers}"
    json_path = output_dir / f"psiformer_attention_benchmark_{tag}.json"
    md_path = output_dir / f"psiformer_attention_benchmark_{tag}.md"
    canonical_json_path = output_dir / "psiformer_attention_benchmark.json"
    canonical_md_path = output_dir / "psiformer_attention_benchmark.md"
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown = _format_markdown(summary)
    json_path.write_text(payload)
    md_path.write_text(markdown)
    canonical_json_path.write_text(payload)
    canonical_md_path.write_text(markdown)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PsiFormer Attention Benchmark",
        "",
        f"Experiment: `{summary['experiment']}`",
        "",
        "| Implementation | Mean s | Median s | Speedup mean | Logabs delta | Finite |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["results"]:
        lines.append(
            "| {implementation} | {mean_seconds:.6f} | {median_seconds:.6f} | "
            "{speedup_vs_first_mean:.3f} | {max_abs_logabs_delta_vs_first:.3e} | "
            "{all_finite} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_platform(platform: str | None) -> str | None:
    """Normalize user-facing platform aliases to JAX platform names."""

    if platform in {None, ""}:
        return None
    if platform == "gpu":
        return "cuda"
    return platform


if __name__ == "__main__":
    raise SystemExit(main())
