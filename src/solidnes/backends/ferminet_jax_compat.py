"""Small JAX compatibility shims for running FermiNet on latest JAX."""

from __future__ import annotations

from typing import Any, Sequence


def apply_modern_jax_shims() -> None:
    """Patch APIs still used by FermiNet/kfac-jax but removed in JAX 0.10."""

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    if not hasattr(jax, "device_put_replicated"):

        def device_put_replicated(obj: Any, devices: Sequence[Any] | None = None) -> Any:
            local_devices = tuple(devices or jax.local_devices())
            sharding = _leading_axis_sharding(jax, local_devices)

            def replicate_leaf(leaf: Any) -> Any:
                stacked = jnp.stack([leaf] * len(local_devices))
                return jax.device_put(stacked, sharding)

            return jax.tree_util.tree_map(replicate_leaf, obj)

        jax.device_put_replicated = device_put_replicated  # type: ignore[attr-defined]

    if not hasattr(jax, "device_put_sharded"):

        def device_put_sharded(shards: Sequence[Any], devices: Sequence[Any] | None = None) -> Any:
            local_devices = tuple(devices or jax.local_devices())
            stacked = jnp.stack(list(shards))
            return jax.device_put(stacked, _leading_axis_sharding(jax, local_devices))

        jax.device_put_sharded = device_put_sharded  # type: ignore[attr-defined]


def _leading_axis_sharding(jax_module: Any, devices: Sequence[Any]) -> Any:
    mesh = jax_module.sharding.Mesh(devices, ("device",))
    return jax_module.sharding.NamedSharding(
        mesh,
        jax_module.sharding.PartitionSpec("device"),
    )

