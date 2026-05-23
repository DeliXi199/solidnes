"""Compatibility shims for running legacy DeepSolid smoke tests."""

from __future__ import annotations

import types


def apply_jax_legacy_shims() -> None:
    """Restore small JAX aliases expected by DeepSolid's legacy KFAC code."""
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import jax._src.core as jax_src_core  # pylint: disable=import-outside-toplevel
    import jax.tree_util as tree_util  # pylint: disable=import-outside-toplevel
    from jax.interpreters import ad as jax_ad  # pylint: disable=import-outside-toplevel

    if not hasattr(jax, "ad"):
        jax.ad = jax_ad
    if not hasattr(jax, "ShapedArray") and hasattr(jax.core, "ShapedArray"):
        jax.ShapedArray = jax.core.ShapedArray
    if not hasattr(jax.core, "lu") and hasattr(jax_src_core, "lu"):
        jax.core.lu = jax_src_core.lu
    if not hasattr(jax.core, "UnitVar"):
        class _SolidnesUnitVar:
            pass

        jax.core.UnitVar = _SolidnesUnitVar
    if not hasattr(jax.core, "unitvar"):
        jax.core.unitvar = jax.core.UnitVar()
    if not hasattr(jax.core, "unit"):
        jax.core.unit = object()
    if not hasattr(jax.core, "extract_call_jaxpr"):
        def _extract_call_jaxpr(primitive, params):
            del primitive
            for key in ("call_jaxpr", "jaxpr"):
                if key in params:
                    call_jaxpr = params[key]
                    if hasattr(call_jaxpr, "jaxpr"):
                        call_jaxpr = call_jaxpr.jaxpr
                    return call_jaxpr, {name: value for name, value in params.items() if name != key}
            return None, params

        jax.core.extract_call_jaxpr = _extract_call_jaxpr

    if not hasattr(jax, "xla"):
        jax.xla = types.SimpleNamespace(translations={})
    elif not hasattr(jax.xla, "translations"):
        jax.xla.translations = {}

    if not hasattr(jax, "tree_multimap"):
        jax.tree_multimap = tree_util.tree_map
    if not hasattr(tree_util, "tree_multimap"):
        tree_util.tree_multimap = tree_util.tree_map
    if not hasattr(jnp, "DeviceArray"):
        jnp.DeviceArray = getattr(jax, "Array", object)


def neutralize_kfac_tags_for_smoke() -> None:
    """Turn KFAC tagging primitives into identity functions for smoke tests.

    DeepSolid's network forward path registers KFAC tags even when the selected
    smoke run is not using KFAC. The tags are not needed for zero-iteration
    runtime smoke tests, and old primitive lowering hooks are incompatible with
    modern JAX. This shim must not be used for production KFAC training.
    """
    from DeepSolid import curvature_tags_and_blocks  # pylint: disable=import-outside-toplevel
    from DeepSolid.utils.kfac_ferminet_alpha import loss_functions  # pylint: disable=import-outside-toplevel

    def _identity_y(y, *args, **kwargs):
        del args, kwargs
        return y

    curvature_tags_and_blocks.register_qmc1 = _identity_y
    curvature_tags_and_blocks.register_qmc2 = _identity_y
    curvature_tags_and_blocks.register_repeated_dense = _identity_y
    loss_functions.register_normal_predictive_distribution = _identity_y
    loss_functions.register_squared_error_loss = _identity_y


def patch_kfac_tag_primitives_for_modern_jax() -> None:
    """Patch legacy KFAC tag primitives for JAX 0.4 abstract evaluation.

    DeepSolid vendors an older KFAC/FermiNet tag implementation whose
    ``Primitive.abstract_eval`` methods return bare avals. JAX 0.4 expects
    primitive abstract evaluation to return ``(avals, effects)`` unless the
    primitive was registered through ``def_abstract_eval``. Without this patch,
    KFAC initialization fails while tracing local-energy derivatives.
    """
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from jax import core as jax_core  # pylint: disable=import-outside-toplevel
    from DeepSolid.utils.kfac_ferminet_alpha import layers_and_loss_tags as tags  # pylint: disable=import-outside-toplevel
    from DeepSolid.utils.kfac_ferminet_alpha import tag_graph_matcher as tgm  # pylint: disable=import-outside-toplevel
    from DeepSolid.utils.kfac_ferminet_alpha import utils as kfac_utils  # pylint: disable=import-outside-toplevel

    if getattr(tags, "_solidnes_modern_jax_tags_patched", False):
        return

    no_effects = getattr(jax_core, "no_effects", frozenset())

    def _layer_abstract_eval(self, *abstract_operands, **kwargs):
        return self.get_outputs(*abstract_operands, **kwargs), no_effects

    def _loss_abstract_eval(self, *args, weight: float, return_loss: bool, **kwargs):
        return self.get_outputs(*args, weight=weight, return_loss=return_loss, **kwargs), no_effects

    def _loss_batching(self, batched_args, batched_dims, **kwargs):
        outputs = self.bind(*batched_args, **kwargs)
        if isinstance(outputs, tuple):
            return outputs, (batched_dims[0],) * len(outputs)
        return outputs, batched_dims[0]

    tags.LayerTag.abstract_eval = _layer_abstract_eval
    tags.LossTag.abstract_eval = _loss_abstract_eval
    tags.LossTag.batching = _loss_batching

    def _psd_inv_cholesky(matrix, damping):
        assert matrix.ndim == 2
        identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
        matrix = matrix + damping * identity
        return kfac_utils.linalg.solve(matrix, identity, assume_a="pos")

    kfac_utils.psd_inv_cholesky = _psd_inv_cholesky
    try:
        from DeepSolid import curvature_tags_and_blocks  # pylint: disable=import-outside-toplevel

        curvature_tags_and_blocks.vmap_psd_inv_cholesky = jax.vmap(_psd_inv_cholesky, (0, None), 0)
    except ImportError:
        pass

    def _evaluate_eqn(eqn, in_values, write_func):
        in_values = list(in_values)
        subfuns, params = eqn.primitive.get_bind_params(eqn.params)
        ans = eqn.primitive.bind(*(list(subfuns) + in_values), **params)
        if eqn.primitive.multiple_results:
            for var, value in zip(eqn.outvars, ans):
                write_func(var, value)
        else:
            write_func(eqn.outvars[0], ans)
        return ans

    tgm.evaluate_eqn = _evaluate_eqn

    try:
        from jax.interpreters import mlir  # pylint: disable=import-outside-toplevel
    except ImportError:
        mlir = None

    if mlir is not None:
        def _layer_lowering(ctx, *operands, **kwargs):
            del ctx, kwargs
            return [operands[0]]

        def _loss_lowering(ctx, *operands, weight, return_loss, **kwargs):
            del ctx, weight, kwargs
            if return_loss:
                raise NotImplementedError("LossTag lowering with return_loss=True is not supported")
            return list(operands)

        # The class methods above fix tracing for existing instances. The MLIR
        # rules are installed lazily per primitive instance in case KFAC later
        # jits a tagged function instead of only inspecting its jaxpr.
        original_layer_init = tags.LayerTag.__init__
        original_loss_init = tags.LossTag.__init__

        def _patched_layer_init(self, *args, **kwargs):
            original_layer_init(self, *args, **kwargs)
            mlir.register_lowering(self, _layer_lowering)

        def _patched_loss_init(self, *args, **kwargs):
            original_loss_init(self, *args, **kwargs)
            mlir.register_lowering(self, _loss_lowering)

        tags.LayerTag.__init__ = _patched_layer_init
        tags.LossTag.__init__ = _patched_loss_init

        def _register_existing_lowerings():
            import gc  # pylint: disable=import-outside-toplevel

            for obj in gc.get_objects():
                if isinstance(obj, tags.LayerTag):
                    mlir.register_lowering(obj, _layer_lowering)
                elif isinstance(obj, tags.LossTag):
                    mlir.register_lowering(obj, _loss_lowering)

        _register_existing_lowerings()

    tags._solidnes_modern_jax_tags_patched = True


def patch_pretrain_kfac_tags_for_modern_jax() -> None:
    """Run DeepSolid pretraining with neutral tags, then restore KFAC tags.

    The legacy KFAC layer tags are required for KFAC optimization, but they are
    not needed for the Hartree-Fock pretraining objective. On the current JAX
    stack, those tags can fail inside the pmapped pretrain path, so this patch
    limits neutralization to pretraining only.
    """
    from DeepSolid import pretrain  # pylint: disable=import-outside-toplevel

    if getattr(pretrain, "_solidnes_kfac_pretrain_tags_patched", False):
        return

    original_net_pretrain = pretrain.pretrain_hartree_fock
    original_hf_pretrain = pretrain.pretrain_hartree_fock_usingHF

    def _capture_tag_functions():
        from DeepSolid import curvature_tags_and_blocks  # pylint: disable=import-outside-toplevel
        from DeepSolid.utils.kfac_ferminet_alpha import loss_functions  # pylint: disable=import-outside-toplevel

        return {
            "curvature_tags_and_blocks": curvature_tags_and_blocks,
            "loss_functions": loss_functions,
            "register_qmc1": curvature_tags_and_blocks.register_qmc1,
            "register_qmc2": curvature_tags_and_blocks.register_qmc2,
            "register_repeated_dense": curvature_tags_and_blocks.register_repeated_dense,
            "register_normal_predictive_distribution": loss_functions.register_normal_predictive_distribution,
            "register_squared_error_loss": loss_functions.register_squared_error_loss,
        }

    def _restore_tag_functions(originals):
        curvature_tags_and_blocks = originals["curvature_tags_and_blocks"]
        loss_functions = originals["loss_functions"]
        curvature_tags_and_blocks.register_qmc1 = originals["register_qmc1"]
        curvature_tags_and_blocks.register_qmc2 = originals["register_qmc2"]
        curvature_tags_and_blocks.register_repeated_dense = originals["register_repeated_dense"]
        loss_functions.register_normal_predictive_distribution = originals["register_normal_predictive_distribution"]
        loss_functions.register_squared_error_loss = originals["register_squared_error_loss"]

    def _with_neutral_pretrain_tags(func):
        def _wrapped(*args, **kwargs):
            originals = _capture_tag_functions()
            neutralize_kfac_tags_for_smoke()
            try:
                return func(*args, **kwargs)
            finally:
                _restore_tag_functions(originals)

        return _wrapped

    pretrain.pretrain_hartree_fock = _with_neutral_pretrain_tags(original_net_pretrain)
    pretrain.pretrain_hartree_fock_usingHF = _with_neutral_pretrain_tags(original_hf_pretrain)
    pretrain._solidnes_kfac_pretrain_tags_patched = True


def patch_checkpoint_save_for_smoke() -> None:
    """Make DeepSolid checkpoint I/O tolerate modern JAX/Optax pytrees.

    DeepSolid's historical checkpoint writer passes nested parameter and
    optimizer pytrees directly to ``np.savez``. With the JAX 0.4.30 smoke stack,
    Optax state can contain heterogeneous sequences that NumPy tries to coerce
    into a regular ndarray, failing at the end of a one-step run. For smoke
    tests and validation runs, object-array checkpointing is enough and avoids
    modifying the external DeepSolid checkout.

    The matching restore shim unwraps params/data/move width. Adam optimizer
    state is intentionally reset on restore because DeepSolid's modern-JAX Adam
    restore branch reconstructs the MultiSteps state incorrectly. This is still
    enough for validation continuation of walkers and parameters.
    """
    import os  # pylint: disable=import-outside-toplevel

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    from DeepSolid import checkpoint  # pylint: disable=import-outside-toplevel
    from DeepSolid import constants  # pylint: disable=import-outside-toplevel

    def _object_array(value):
        array = np.empty((), dtype=object)
        array[()] = value
        return array

    def _save_for_smoke(save_path: str, t: int, data, params, opt_state, mcmc_width, remote_save_path=None) -> str:
        del remote_save_path
        ckpt_filename = os.path.join(save_path, f"qmcjax_ckpt_{t:06d}.npz")
        with open(ckpt_filename, "wb") as handle:
            np.savez(
                handle,
                t=t,
                data=_object_array(data),
                params=_object_array(params),
                opt_state=_object_array(opt_state),
                mcmc_width=_object_array(mcmc_width),
            )
        return ckpt_filename

    def _unwrap_object_array(value):
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            return value.item()
        return value

    def _reshard_data(data, batch_size):
        num_devices = jax.local_device_count()
        if batch_size is None:
            return data
        if batch_size % num_devices != 0:
            raise ValueError(
                "Batch size must be divisible by number of devices after restore, "
                f"got batch size {batch_size} for {num_devices} devices."
            )
        saved_batch_size = data.shape[0] * data.shape[1]
        if saved_batch_size != batch_size:
            raise ValueError(
                "Wrong batch size in loaded data. Expected {}, found {}.".format(
                    batch_size, saved_batch_size
                )
            )
        if data.shape[0] == num_devices:
            return data
        flattened = jnp.reshape(jnp.asarray(data), (batch_size,) + tuple(data.shape[2:]))
        return jnp.reshape(flattened, (num_devices, batch_size // num_devices) + tuple(data.shape[2:]))

    def _looks_replicated(value, saved_devices):
        return hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == saved_devices

    def _reshard_replicated_tree(tree, saved_devices):
        num_devices = jax.local_device_count()
        if saved_devices == num_devices:
            return tree
        unreplicated = jax.tree_map(
            lambda value: value[0] if _looks_replicated(value, saved_devices) else value,
            tree,
        )
        return constants.replicate_all_local_devices(unreplicated)

    def _reshard_mcmc_width(mcmc_width, saved_devices):
        num_devices = jax.local_device_count()
        if mcmc_width is None or saved_devices == num_devices:
            return mcmc_width
        if _looks_replicated(mcmc_width, saved_devices):
            mcmc_width = mcmc_width[0]
        return jnp.stack([jnp.asarray(mcmc_width)] * num_devices, axis=0)

    def _restore_for_smoke(restore_filename: str, batch_size=None, shape_check=True):
        with open(restore_filename, "rb") as handle:
            ckpt_data = np.load(handle, allow_pickle=True)
            t = ckpt_data["t"].tolist() + 1
            data = _unwrap_object_array(ckpt_data["data"])
            params = _unwrap_object_array(ckpt_data["params"])
            mcmc_width = _unwrap_object_array(ckpt_data["mcmc_width"])
            if shape_check:
                saved_devices = data.shape[0]
                data = _reshard_data(data, batch_size)
                params = _reshard_replicated_tree(params, saved_devices)
                mcmc_width = _reshard_mcmc_width(mcmc_width, saved_devices)
        return t, data, params, None, mcmc_width

    checkpoint.save = _save_for_smoke
    checkpoint.restore = _restore_for_smoke
