#!/usr/bin/env python
"""Check SolidNES paper-aligned native FermiNet overlap-loss helpers."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"
for path in (PROJECT_ROOT / "src", FERMINET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp
import numpy as np

from ferminet import loss as ferminet_loss
from ferminet import networks
from ferminet import train as ferminet_train
from solidnes.backends.ferminet_adapter import build_ferminet_adapter


def main() -> int:
    _check_symmetrized_squared_overlap()
    _check_overlap_gradient_scale()
    _check_overlap_gradient_scale_uses_ewm_inputs()
    _check_deepqmc_style_overlap_ewm()
    _check_tree_all_finite()
    _check_spin_penalty_energy_helper()
    _check_spin_penalty_bare_energy_helper()
    _check_ratio_clipping()
    _check_state_ordering()
    _check_custom_jvp_smoke()
    _check_loss_consumes_ewm_scale_fields()
    _check_method_profile_adapter_defaults()
    _check_spin_penalty_adapter_plumbing()
    print("ferminet_native_overlap_loss_alignment: ok")
    return 0


def _check_symmetrized_squared_overlap() -> None:
    overlap = jnp.array([[1.0, -0.2], [0.3, 1.0]])
    squared = ferminet_loss._symmetrized_squared_overlap(overlap)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(squared[0, 1]), 0.0)
    symmetric = ferminet_loss._symmetrized_overlap(overlap)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(symmetric[0, 1]), 0.0)
    positive = jnp.array([[1.0, 0.25], [0.5, 1.0]])
    squared = ferminet_loss._symmetrized_squared_overlap(positive)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(squared[0, 1]), 0.125)
    symmetric = ferminet_loss._symmetrized_overlap(positive)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(symmetric[0, 1]), np.sqrt(0.125))


def _check_overlap_gradient_scale() -> None:
    energy = jnp.array([-2.0, -2.25])
    variance = jnp.array([0.01, 0.0004])
    scale = ferminet_loss._overlap_gradient_scale(  # pylint: disable=protected-access
        energy,
        variance,
        "max_gap_std",
        0.001,
        5.0,
    )
    expected = np.array([[0.1, 0.25], [0.25, 0.02]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(scale), expected, rtol=1e-6)


def _check_overlap_gradient_scale_uses_ewm_inputs() -> None:
    batch_energy = jnp.array([-10.0, -20.0])
    batch_variance = jnp.array([100.0, 100.0])
    scale_energy = jnp.array([-2.0, -2.25])
    scale_std = jnp.array([0.1, 0.02])
    scale = ferminet_loss._overlap_gradient_scale(  # pylint: disable=protected-access
        batch_energy,
        batch_variance,
        "max_gap_std",
        0.001,
        5.0,
        scale_energy=scale_energy,
        scale_std=scale_std,
    )
    expected = np.array([[0.1, 0.25], [0.25, 0.02]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(scale), expected, rtol=1e-6)


def _check_deepqmc_style_overlap_ewm() -> None:
    state = ferminet_train._init_overlap_scale_ewm_state(  # pylint: disable=protected-access
        2,
        max_alpha=0.999,
        decay_alpha=10.0,
        window_size=4,
    )

    class Aux:
        energy = jnp.array([10.0, 20.0], dtype=jnp.float32)
        variance = jnp.array([4.0, 9.0], dtype=jnp.float32)

    state = ferminet_train._update_overlap_scale_ewm(  # pylint: disable=protected-access
        Aux(),
        state,
        max_alpha=0.999,
        decay_alpha=10.0,
    )
    np.testing.assert_allclose(
        np.asarray(state.energy_mean),
        np.array([5.0, 10.0], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state.std_mean),
        np.array([1.0, 1.5], dtype=np.float32),
        rtol=1e-6,
    )

    class Aux2:
        energy = jnp.array([14.0, 22.0], dtype=jnp.float32)
        variance = jnp.array([1.0, 16.0], dtype=jnp.float32)

    state = ferminet_train._update_overlap_scale_ewm(  # pylint: disable=protected-access
        Aux2(),
        state,
        max_alpha=0.999,
        decay_alpha=10.0,
    )
    second_alpha = 1.0 / (2.0 + 1.0 / 10.0)
    first_alpha = 0.5
    expected_weights = np.array(
        [second_alpha, first_alpha * (1.0 - second_alpha)], dtype=np.float32
    )
    expected_energy = (
        expected_weights[:, None]
        * np.array([[14.0, 22.0], [10.0, 20.0]], dtype=np.float32)
    ).sum(axis=0)
    expected_std = (
        expected_weights[:, None]
        * np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32)
    ).sum(axis=0)
    np.testing.assert_allclose(np.asarray(state.energy_mean), expected_energy, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state.std_mean), expected_std, rtol=1e-6)


def _check_tree_all_finite() -> None:
    finite_tree = {
        "params": [jnp.array([1.0, 2.0]), jnp.array([3], dtype=jnp.int32)],
        "none": None,
    }
    if not bool(ferminet_train._tree_all_finite(finite_tree)):  # pylint: disable=protected-access
        raise AssertionError("finite pytree was reported non-finite")
    bad_tree = {"params": [jnp.array([1.0, jnp.nan])]}
    if bool(ferminet_train._tree_all_finite(bad_tree)):  # pylint: disable=protected-access
        raise AssertionError("non-finite pytree was reported finite")


def _check_spin_penalty_energy_helper() -> None:
    s2 = jnp.array([[0.2, 0.01], [0.01, 2.1]], dtype=jnp.float32)

    state_specific_energy, state_specific_aux = (
        ferminet_train._apply_spin_penalty_to_local_energy(  # pylint: disable=protected-access
            jnp.array([-2.0, -1.0], dtype=jnp.float32),
            None,
            s2,
            0.5,
            2,
        )
    )
    np.testing.assert_allclose(
        np.asarray(state_specific_energy),
        np.array([-1.9, 0.05], dtype=np.float32),
        rtol=1e-6,
    )
    if state_specific_aux is not None:
        raise AssertionError("state-specific spin penalty should preserve None aux")

    matrix_energy, matrix_aux = ferminet_train._apply_spin_penalty_to_local_energy(  # pylint: disable=protected-access
        jnp.array(-3.0, dtype=jnp.float32),
        jnp.eye(2, dtype=jnp.float32),
        s2,
        0.5,
        2,
    )
    np.testing.assert_allclose(np.asarray(matrix_energy), -1.85, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(matrix_aux),
        np.eye(2, dtype=np.float32) + 0.5 * np.asarray(s2),
        rtol=1e-6,
    )


def _check_spin_penalty_bare_energy_helper() -> None:
    s2 = jnp.array([[0.2, 0.01], [0.01, 2.1]], dtype=jnp.float32)
    state_energy = jnp.array([-1.9, 0.05], dtype=jnp.float32)
    bare_state_energy = ferminet_train._remove_spin_penalty_from_energy_matrix(  # pylint: disable=protected-access
        state_energy,
        s2,
        0.5,
    )
    np.testing.assert_allclose(
        np.asarray(bare_state_energy),
        np.array([-2.0, -1.0], dtype=np.float32),
        rtol=1e-6,
    )

    energy_matrix = jnp.eye(2, dtype=jnp.float32) + 0.5 * s2
    bare_energy_matrix = ferminet_train._remove_spin_penalty_from_energy_matrix(  # pylint: disable=protected-access
        energy_matrix,
        s2,
        0.5,
    )
    np.testing.assert_allclose(
        np.asarray(bare_energy_matrix),
        np.eye(2, dtype=np.float32),
        rtol=1e-6,
    )


def _check_ratio_clipping() -> None:
    ratios = jnp.array(
        [
            [[1.0, 0.1], [0.2, 1.0]],
            [[1.0, 0.2], [0.3, 1.0]],
            [[1.0, 0.15], [0.25, 1.0]],
            [[1.0, 100.0], [0.2, 1.0]],
        ],
        dtype=jnp.float32,
    )
    clipped, mask = ferminet_loss._clip_overlap_ratios(  # pylint: disable=protected-access
        ratios,
        clip_width=10.0,
        exclude_width=float("inf"),
    )
    if not bool(jnp.all(jnp.isfinite(clipped))):
        raise AssertionError("clipped overlap ratios must be finite")
    if clipped[-1, 0, 1] >= ratios[-1, 0, 1]:
        raise AssertionError("large ratio outlier was not clipped")
    if not bool(jnp.all(mask)):
        raise AssertionError("infinite exclude_width should keep all mask entries")


def _check_state_ordering() -> None:
    energy = jnp.array([-1.0, -3.0, -2.0])
    ordering = ferminet_loss._state_ordering(energy, "energy")  # pylint: disable=protected-access
    np.testing.assert_array_equal(np.asarray(ordering), np.array([1, 2, 0]))
    vector = jnp.array([[10.0, 20.0, 30.0]])
    ordered = jnp.take(vector, ordering, axis=-1)
    restored = ferminet_loss._unpermute_state_vector(ordered, ordering)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(restored), np.asarray(vector))

    two_state_order = ferminet_loss._state_ordering(  # pylint: disable=protected-access
        jnp.array([-1.0, -3.0]),
        "energy",
    )
    clipped_overlap = jnp.array([[[1.0, 2.0], [3.0, 1.0]]])
    centered_overlap_diff = jnp.array([[[0.0, 5.0], [7.0, 0.0]]])
    scale = jnp.ones((2, 2), dtype=jnp.float32)
    ordered_clipped = ferminet_loss._permute_state_matrix(  # pylint: disable=protected-access
        clipped_overlap,
        two_state_order,
    )
    ordered_diff = ferminet_loss._permute_state_matrix(  # pylint: disable=protected-access
        centered_overlap_diff,
        two_state_order,
    )
    ordered_scale = ferminet_loss._permute_state_matrix(scale, two_state_order)  # pylint: disable=protected-access
    ordered_pair_diff = ordered_clipped * ordered_diff.transpose((0, 2, 1))
    ordered_overlap_diff = 2.0 * jnp.sum(
        jnp.triu(ordered_pair_diff * ordered_scale, 1),
        axis=1,
    )
    original_overlap_diff = ferminet_loss._unpermute_state_vector(  # pylint: disable=protected-access
        ordered_overlap_diff,
        two_state_order,
    )
    np.testing.assert_allclose(np.asarray(original_overlap_diff), np.array([[30.0, 0.0]]))


def _check_custom_jvp_smoke() -> None:
    import jax

    def network(params, positions, spins, atoms, charges):  # noqa: ARG001
        del spins, atoms, charges
        x = positions[0]
        log_psi = params[:, None] + jnp.array([[0.0, 0.2], [-0.1, 0.0]]) * x
        return jnp.ones_like(log_psi), log_psi

    def local_energy(params, key, data):  # noqa: ARG001
        del params, key
        x = data.positions[0]
        return jnp.array([1.0 + x, 2.0 + 0.5 * x]), None

    data = networks.FermiNetData(
        positions=jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32),
        spins=jnp.zeros((4, 1), dtype=jnp.float32),
        atoms=jnp.zeros((4, 1, 3), dtype=jnp.float32),
        charges=jnp.ones((4, 1), dtype=jnp.float32),
    )
    loss_fn = ferminet_loss.make_energy_overlap_loss(
        network,
        local_energy,
        clip_local_energy=5.0,
        clip_from_median=True,
        center_at_clipped_energy=True,
        overlap_penalty=4.0,
        overlap_weight=(0.5, 0.5),
        overlap_scale_by="max_gap_std",
        overlap_min_scale=0.001,
        overlap_max_scale=5.0,
        overlap_clip_width=10.0,
        overlap_sort_states_by="energy",
    )
    params = jnp.array([0.1, -0.2], dtype=jnp.float32)
    value, grad = jax.value_and_grad(lambda p: loss_fn(p, jax.random.PRNGKey(0), data)[0])(
        params
    )
    if not bool(jnp.isfinite(value)):
        raise AssertionError("custom-JVP overlap loss must be finite")
    if not bool(jnp.all(jnp.isfinite(grad))):
        raise AssertionError("custom-JVP overlap loss gradient must be finite")


def _check_loss_consumes_ewm_scale_fields() -> None:
    def network(params, positions, spins, atoms, charges):  # noqa: ARG001
        del params, spins, atoms, charges
        x = positions[0]
        log_psi = jnp.array([[0.0, 0.2], [-0.1, 0.0]]) * x
        return jnp.ones_like(log_psi), log_psi

    def local_energy(params, key, data):  # noqa: ARG001
        del params, key
        x = data.positions[0]
        return jnp.array([100.0 + x, -100.0 - x]), None

    data = networks.FermiNetData(
        positions=jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32),
        spins=jnp.zeros((4, 1), dtype=jnp.float32),
        atoms=jnp.zeros((4, 1, 3), dtype=jnp.float32),
        charges=jnp.ones((4, 1), dtype=jnp.float32),
        overlap_scale_energy=jnp.array([-2.0, -2.25], dtype=jnp.float32),
        overlap_scale_std=jnp.array([0.1, 0.02], dtype=jnp.float32),
        overlap_state_order_energy=jnp.array([-1.0, -3.0], dtype=jnp.float32),
    )
    loss_fn = ferminet_loss.make_energy_overlap_loss(
        network,
        local_energy,
        overlap_weight=(0.5, 0.5),
        overlap_scale_by="max_gap_std",
        overlap_min_scale=0.001,
        overlap_max_scale=5.0,
        overlap_sort_states_by="energy",
    )
    _, aux = loss_fn(jnp.array([0.0], dtype=jnp.float32), jax_random_key(), data)
    expected_scale = np.array([[0.1, 0.25], [0.25, 0.02]], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(aux.overlap_gradient_scale), expected_scale, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(aux.state_ordering), np.array([1, 0]))


def jax_random_key():
    import jax

    return jax.random.PRNGKey(0)


def _check_method_profile_adapter_defaults() -> None:
    bundle = build_ferminet_adapter(
        PROJECT_ROOT
        / "configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned.yaml"
    )
    summary = bundle.summary
    if summary.method_profile != "szabo_noe_2024_penalty":
        raise AssertionError("method profile was not propagated")
    if not summary.overlap_use_ewm_scale:
        raise AssertionError("paper method profile must enable EWM overlap scaling")
    np.testing.assert_allclose(summary.kfac_norm_constraint, 0.002)
    if not summary.kfac_norm_constraint_scale_by_states:
        raise AssertionError("KFAC norm constraint must be state-count scaled")
    np.testing.assert_allclose(summary.spin_penalty, 0.0)
    if summary.s2_observable:
        raise AssertionError("base paper-aligned config should not enable S^2")


def _check_spin_penalty_adapter_plumbing() -> None:
    bundle = build_ferminet_adapter(
        PROJECT_ROOT
        / "configs/experiment/diamond_c_ferminet_pbc_gamma_native_vmc_overlap_kfac_paper_aligned_spin_smoke.yaml"
    )
    summary = bundle.summary
    np.testing.assert_allclose(summary.spin_penalty, 0.1)
    np.testing.assert_allclose(float(bundle.cfg.optim.spin_energy), 0.1)
    if not summary.s2_observable:
        raise AssertionError("spin smoke config must enable S^2 diagnostics")
    if not bool(bundle.cfg.observables.s2):
        raise AssertionError("spin smoke config did not set cfg.observables.s2")


if __name__ == "__main__":
    raise SystemExit(main())
