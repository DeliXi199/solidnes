"""Adapter helpers for building FermiNet configs from SolidNES YAML."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import ml_collections

from solidnes.backends.deepsolid_adapter import load_yaml
from solidnes.backends.deepsolid_adapter import resolve_config_path
from solidnes.backends.ferminet_psiformer_attention import (
    psiformer_attention_implementation,
)
from solidnes.excited_state_mainline import classify_excited_state_mainline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FERMINET_ROOT = PROJECT_ROOT / "external" / "ferminet"


@dataclass(frozen=True)
class FermiNetAdapterPaths:
    """Resolved config paths used to build a FermiNet config."""

    experiment: Path
    system: Path
    model: Path
    sampler: Path
    train: Path


@dataclass(frozen=True)
class FermiNetAdapterSummary:
    """Small serializable summary of the adapter output."""

    experiment_name: str
    backend: str
    config_module: str
    nelec: tuple[int, int]
    natoms: int
    atoms: list[str]
    lattice_bohr: list[list[float]] | None
    kpoints: int | None
    batch_size: int
    optimizer: str
    objective: str
    method_profile: str | None
    excited_state_route: str | None
    excited_state_route_role: str | None
    excited_state_route_is_mainline: bool
    states: int
    independent_state_params: bool
    independent_state_merge_keys: tuple[str, ...]
    diagonal_mcmc_trace: bool
    diagonal_local_energy: bool
    diagonal_overlap_jvp: bool
    profile_step_times: bool
    profile_loss_components: bool
    overlap_penalty: float
    overlap_weights: tuple[float, ...] | None
    overlap_scale_by: str | None
    overlap_min_scale: float
    overlap_max_scale: float
    overlap_clip_width: float
    overlap_clip_exclude_width: float
    overlap_sort_states_by: str | None
    overlap_use_ewm_scale: bool
    overlap_ewm_max_alpha: float
    overlap_ewm_decay_alpha: float
    fixed_ground_checkpoint: str | None
    fixed_ground_init_trainable_from_checkpoint: bool
    fixed_ground_init_trainable_noise_scale: float | None
    fixed_ground_overlap_penalty: float | None
    fixed_ground_overlap_clip_width: float | None
    fixed_ground_overlap_clip_exclude_width: float | None
    fixed_ground_overlap_gradient_scale: float | None
    fixed_ground_symmetric_sampling: bool
    fixed_ground_energy_reference: float | None
    spin_penalty: float
    log_spin_by_state: bool
    s2_observable: bool
    check_nan: bool
    reset_if_nan: bool
    kfac_norm_constraint: float | None
    kfac_norm_constraint_scale_by_states: bool
    iterations: int
    pretrain_method: str | None
    pretrain_iterations: int
    pretrain_target_backend: str
    pretrain_jax_pbc_image_cutoff: int
    pretrain_log_every: int
    pretrain_target_chunk_size: int
    mcmc_burn_in: int
    mcmc_steps_per_iteration: int
    network_type: str
    determinants: int
    hidden_dims: Any
    full_det: bool
    complex_output: bool
    laplacian: str
    forward_laplacian_enabled: bool
    psiformer_tf32: bool | None
    psiformer_attention_implementation: str | None
    psiformer_attention_kernel_gpu: str | None
    target_jax_version: str
    precision_profile: str
    x64_enabled: bool
    save_path: str
    restore_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "backend": self.backend,
            "config_module": self.config_module,
            "nelec": self.nelec,
            "natoms": self.natoms,
            "atoms": self.atoms,
            "lattice_bohr": self.lattice_bohr,
            "kpoints": self.kpoints,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "objective": self.objective,
            "method_profile": self.method_profile,
            "excited_state_route": self.excited_state_route,
            "excited_state_route_role": self.excited_state_route_role,
            "excited_state_route_is_mainline": self.excited_state_route_is_mainline,
            "states": self.states,
            "independent_state_params": self.independent_state_params,
            "independent_state_merge_keys": self.independent_state_merge_keys,
            "diagonal_mcmc_trace": self.diagonal_mcmc_trace,
            "diagonal_local_energy": self.diagonal_local_energy,
            "diagonal_overlap_jvp": self.diagonal_overlap_jvp,
            "profile_step_times": self.profile_step_times,
            "profile_loss_components": self.profile_loss_components,
            "overlap_penalty": self.overlap_penalty,
            "overlap_weights": self.overlap_weights,
            "overlap_scale_by": self.overlap_scale_by,
            "overlap_min_scale": self.overlap_min_scale,
            "overlap_max_scale": self.overlap_max_scale,
            "overlap_clip_width": self.overlap_clip_width,
            "overlap_clip_exclude_width": self.overlap_clip_exclude_width,
            "overlap_sort_states_by": self.overlap_sort_states_by,
            "overlap_use_ewm_scale": self.overlap_use_ewm_scale,
            "overlap_ewm_max_alpha": self.overlap_ewm_max_alpha,
            "overlap_ewm_decay_alpha": self.overlap_ewm_decay_alpha,
            "fixed_ground_checkpoint": self.fixed_ground_checkpoint,
            "fixed_ground_init_trainable_from_checkpoint": (
                self.fixed_ground_init_trainable_from_checkpoint
            ),
            "fixed_ground_init_trainable_noise_scale": (
                self.fixed_ground_init_trainable_noise_scale
            ),
            "fixed_ground_overlap_penalty": self.fixed_ground_overlap_penalty,
            "fixed_ground_overlap_clip_width": self.fixed_ground_overlap_clip_width,
            "fixed_ground_overlap_clip_exclude_width": (
                self.fixed_ground_overlap_clip_exclude_width
            ),
            "fixed_ground_overlap_gradient_scale": (
                self.fixed_ground_overlap_gradient_scale
            ),
            "fixed_ground_symmetric_sampling": self.fixed_ground_symmetric_sampling,
            "fixed_ground_energy_reference": self.fixed_ground_energy_reference,
            "spin_penalty": self.spin_penalty,
            "log_spin_by_state": self.log_spin_by_state,
            "s2_observable": self.s2_observable,
            "check_nan": self.check_nan,
            "reset_if_nan": self.reset_if_nan,
            "kfac_norm_constraint": self.kfac_norm_constraint,
            "kfac_norm_constraint_scale_by_states": self.kfac_norm_constraint_scale_by_states,
            "iterations": self.iterations,
            "pretrain_method": self.pretrain_method,
            "pretrain_iterations": self.pretrain_iterations,
            "pretrain_target_backend": self.pretrain_target_backend,
            "pretrain_jax_pbc_image_cutoff": self.pretrain_jax_pbc_image_cutoff,
            "pretrain_log_every": self.pretrain_log_every,
            "pretrain_target_chunk_size": self.pretrain_target_chunk_size,
            "mcmc_burn_in": self.mcmc_burn_in,
            "mcmc_steps_per_iteration": self.mcmc_steps_per_iteration,
            "network_type": self.network_type,
            "determinants": self.determinants,
            "hidden_dims": self.hidden_dims,
            "full_det": self.full_det,
            "complex_output": self.complex_output,
            "laplacian": self.laplacian,
            "forward_laplacian_enabled": self.forward_laplacian_enabled,
            "psiformer_tf32": self.psiformer_tf32,
            "psiformer_attention_implementation": (
                self.psiformer_attention_implementation
            ),
            "psiformer_attention_kernel_gpu": self.psiformer_attention_kernel_gpu,
            "target_jax_version": self.target_jax_version,
            "precision_profile": self.precision_profile,
            "x64_enabled": self.x64_enabled,
            "save_path": self.save_path,
            "restore_path": self.restore_path,
        }


@dataclass(frozen=True)
class FermiNetAdapterBundle:
    """Built FermiNet config plus the SolidNES inputs that produced it."""

    cfg: Any
    experiment: dict[str, Any]
    system: dict[str, Any]
    model: dict[str, Any]
    sampler: dict[str, Any]
    train: dict[str, Any]
    paths: FermiNetAdapterPaths

    @property
    def summary(self) -> FermiNetAdapterSummary:
        kwargs = self.cfg.system.make_local_energy_kwargs
        lattice = kwargs.get("lattice")
        kpoints = self.cfg.network.make_envelope_kwargs.get("kpoints")
        hidden_dims = (
            self.cfg.network.ferminet.hidden_dims
            if self.cfg.network.network_type == "ferminet"
            else {
                "num_layers": self.cfg.network.psiformer.num_layers,
                "num_heads": self.cfg.network.psiformer.num_heads,
                "heads_dim": self.cfg.network.psiformer.heads_dim,
                "mlp_hidden_dims": self.cfg.network.psiformer.mlp_hidden_dims,
            }
        )
        runtime = self.experiment.get("runtime", {})
        fixed_ground = self.cfg.optim.get("fixed_ground", {})
        psiformer_attention = psiformer_attention_implementation(self.cfg)
        independent_states = bool(self.cfg.network.get("independent_states", False))
        independent_state_merge_keys = tuple(
            str(key)
            for key in self.cfg.network.get("independent_state_merge_keys", ())
        )
        auto_diagonal_paths = (
            independent_states and str(self.cfg.optim.objective) == "vmc_overlap"
        )
        route = classify_excited_state_mainline(
            objective=str(self.cfg.optim.objective),
            states=int(self.cfg.system.get("states", 0)),
            network_type=str(self.cfg.network.network_type),
            attention_implementation=psiformer_attention,
            attention_kernel_gpu=(
                self.cfg.network.get("psiformer_attention_kernel_gpu")
                if self.cfg.network.network_type == "psiformer"
                else None
            ),
            method_profile=self.cfg.optim.get("method_profile"),
            independent_state_params=independent_states,
            independent_state_merge_keys=independent_state_merge_keys,
        )
        return FermiNetAdapterSummary(
            experiment_name=self.experiment["experiment_name"],
            backend=self.experiment["backend"]["name"],
            config_module=self.experiment["ferminet_config_template"]["module"],
            nelec=tuple(int(x) for x in self.cfg.system.electrons),
            natoms=len(self.cfg.system.molecule),
            atoms=[atom.symbol for atom in self.cfg.system.molecule],
            lattice_bohr=lattice.tolist() if lattice is not None else None,
            kpoints=int(kpoints.shape[0]) if kpoints is not None else None,
            batch_size=int(self.cfg.batch_size),
            optimizer=self.cfg.optim.optimizer,
            objective=str(self.cfg.optim.objective),
            method_profile=self.cfg.optim.get("method_profile"),
            excited_state_route=route.method,
            excited_state_route_role=route.role,
            excited_state_route_is_mainline=route.is_mainline,
            states=int(self.cfg.system.get("states", 0)),
            independent_state_params=independent_states,
            independent_state_merge_keys=independent_state_merge_keys,
            diagonal_mcmc_trace=_resolve_auto_bool(
                self.cfg.optim.get("diagonal_mcmc_trace", None),
                auto_diagonal_paths,
            ),
            diagonal_local_energy=_resolve_auto_bool(
                self.cfg.optim.get("diagonal_local_energy", None),
                auto_diagonal_paths,
            ),
            diagonal_overlap_jvp=_resolve_auto_bool(
                self.cfg.optim.get("diagonal_overlap_jvp", None),
                auto_diagonal_paths,
            ),
            profile_step_times=bool(self.cfg.log.get("profile_step_times", False)),
            profile_loss_components=bool(
                self.cfg.log.get("profile_loss_components", False)
            ),
            overlap_penalty=float(self.cfg.optim.overlap.penalty),
            overlap_weights=None
            if self.cfg.optim.overlap.weights is None
            else tuple(float(x) for x in self.cfg.optim.overlap.weights),
            overlap_scale_by=self.cfg.optim.overlap.get("scale_by"),
            overlap_min_scale=float(self.cfg.optim.overlap.get("min_scale", 0.001)),
            overlap_max_scale=float(self.cfg.optim.overlap.get("max_scale", 5.0)),
            overlap_clip_width=float(self.cfg.optim.overlap.get("clip_width", 10.0)),
            overlap_clip_exclude_width=float(
                self.cfg.optim.overlap.get("clip_exclude_width", float("inf"))
            ),
            overlap_sort_states_by=self.cfg.optim.overlap.get("sort_states_by"),
            overlap_use_ewm_scale=bool(
                self.cfg.optim.overlap.get("use_ewm_scale", False)
            ),
            overlap_ewm_max_alpha=float(
                self.cfg.optim.overlap.get("ewm_max_alpha", 0.999)
            ),
            overlap_ewm_decay_alpha=float(
                self.cfg.optim.overlap.get("ewm_decay_alpha", 10.0)
            ),
            fixed_ground_checkpoint=fixed_ground.get("checkpoint"),
            fixed_ground_init_trainable_from_checkpoint=bool(
                fixed_ground.get("init_trainable_from_checkpoint", False)
            ),
            fixed_ground_init_trainable_noise_scale=(
                float(fixed_ground.get("init_trainable_noise_scale"))
                if fixed_ground.get("init_trainable_noise_scale") is not None
                else None
            ),
            fixed_ground_overlap_penalty=(
                float(fixed_ground.get("overlap_penalty"))
                if fixed_ground.get("overlap_penalty") is not None
                else None
            ),
            fixed_ground_overlap_clip_width=(
                float(fixed_ground.get("clip_width"))
                if fixed_ground.get("clip_width") is not None
                else None
            ),
            fixed_ground_overlap_clip_exclude_width=(
                float(fixed_ground.get("clip_exclude_width"))
                if fixed_ground.get("clip_exclude_width") is not None
                else None
            ),
            fixed_ground_overlap_gradient_scale=(
                float(fixed_ground.get("gradient_scale"))
                if fixed_ground.get("gradient_scale") is not None
                else None
            ),
            fixed_ground_symmetric_sampling=bool(
                fixed_ground.get("symmetric_sampling", False)
            ),
            fixed_ground_energy_reference=(
                float(fixed_ground.get("energy_reference"))
                if fixed_ground.get("energy_reference") is not None
                else None
            ),
            spin_penalty=float(self.cfg.optim.get("spin_energy", 0.0)),
            log_spin_by_state=bool(self.cfg.optim.get("log_spin_by_state", False)),
            s2_observable=bool(self.cfg.observables.get("s2", False)),
            check_nan=bool(self.cfg.debug.get("check_nan", False)),
            reset_if_nan=bool(self.cfg.optim.get("reset_if_nan", False)),
            kfac_norm_constraint=(
                float(self.cfg.optim.kfac.norm_constraint)
                if self.cfg.optim.optimizer == "kfac"
                else None
            ),
            kfac_norm_constraint_scale_by_states=bool(
                self.cfg.optim.kfac.get("norm_constraint_scale_by_states", False)
            )
            if self.cfg.optim.optimizer == "kfac"
            else False,
            iterations=int(self.cfg.optim.iterations),
            pretrain_method=self.cfg.pretrain.method,
            pretrain_iterations=int(self.cfg.pretrain.iterations),
            pretrain_target_backend=str(
                self.cfg.pretrain.get("target_backend", "pyscf_pbc")
            ),
            pretrain_jax_pbc_image_cutoff=int(
                self.cfg.pretrain.get("jax_pbc_image_cutoff", 2)
            ),
            pretrain_log_every=int(self.cfg.pretrain.get("log_every", 1)),
            pretrain_target_chunk_size=int(
                self.cfg.pretrain.get("target_chunk_size", 0)
            ),
            mcmc_burn_in=int(self.cfg.mcmc.burn_in),
            mcmc_steps_per_iteration=int(self.cfg.mcmc.steps),
            network_type=self.cfg.network.network_type,
            determinants=int(self.cfg.network.determinants),
            hidden_dims=hidden_dims,
            full_det=bool(self.cfg.network.full_det),
            complex_output=bool(self.cfg.network.get("complex", False)),
            laplacian=self.cfg.optim.laplacian,
            forward_laplacian_enabled=self.cfg.optim.laplacian == "folx",
            psiformer_tf32=(
                bool(self.cfg.network.psiformer.tf32)
                if self.cfg.network.network_type == "psiformer"
                else None
            ),
            psiformer_attention_implementation=psiformer_attention,
            psiformer_attention_kernel_gpu=(
                self.cfg.network.get("psiformer_attention_kernel_gpu")
                if self.cfg.network.network_type == "psiformer"
                else None
            ),
            target_jax_version=str(runtime.get("target_jax_version", "latest")),
            precision_profile=str(runtime.get("precision_profile", "speed")),
            x64_enabled=bool(runtime.get("x64_enabled", False)),
            save_path=self.cfg.log.save_path,
            restore_path=self.cfg.log.restore_path,
        )


def build_ferminet_adapter(
    experiment_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    ferminet_root: Path = DEFAULT_FERMINET_ROOT,
) -> FermiNetAdapterBundle:
    """Build a FermiNet config and retain SolidNES-side metadata."""

    _ensure_import_paths(project_root, ferminet_root)
    resolved_experiment_path = _resolve_project_path(experiment_path, project_root)
    experiment = load_yaml(resolved_experiment_path)
    paths = FermiNetAdapterPaths(
        experiment=resolved_experiment_path,
        system=resolve_config_path(resolved_experiment_path, experiment["system_config"]),
        model=resolve_config_path(resolved_experiment_path, experiment["model_config"]),
        sampler=resolve_config_path(resolved_experiment_path, experiment["sampler_config"]),
        train=resolve_config_path(resolved_experiment_path, experiment["train_config"]),
    )

    system_cfg = load_yaml(paths.system)
    model_cfg = load_yaml(paths.model)
    sampler_cfg = load_yaml(paths.sampler)
    train_cfg = load_yaml(paths.train)

    cfg = _build_ferminet_config(
        experiment, system_cfg, model_cfg, sampler_cfg, train_cfg, project_root
    )
    return FermiNetAdapterBundle(
        cfg=cfg,
        experiment=experiment,
        system=system_cfg,
        model=model_cfg,
        sampler=sampler_cfg,
        train=train_cfg,
        paths=paths,
    )


def create_output_dirs(bundle: FermiNetAdapterBundle) -> None:
    """Create output directories referenced by the FermiNet config."""

    Path(bundle.cfg.log.save_path).mkdir(parents=True, exist_ok=True)
    output = bundle.experiment.get("output", {})
    log_dir = output.get("log_dir")
    if log_dir:
        Path(PROJECT_ROOT / log_dir).mkdir(parents=True, exist_ok=True)


def format_summary(summary: FermiNetAdapterSummary) -> str:
    """Format a FermiNet adapter summary for CLI diagnostics."""

    lines = [
        f"experiment: {summary.experiment_name}",
        f"backend: {summary.backend}",
        f"config_module: {summary.config_module}",
        "",
        f"natoms: {summary.natoms}",
        f"atoms: {summary.atoms}",
        f"nelec: {summary.nelec}",
        f"kpoints: {summary.kpoints}",
        "",
        f"network_type: {summary.network_type}",
        f"determinants: {summary.determinants}",
        f"hidden_dims: {summary.hidden_dims}",
        f"full_det: {summary.full_det}",
        f"complex_output: {summary.complex_output}",
        "",
        f"optimizer: {summary.optimizer}",
        f"objective: {summary.objective}",
        f"method_profile: {summary.method_profile}",
        f"excited_state_route: {summary.excited_state_route}",
        f"excited_state_route_role: {summary.excited_state_route_role}",
        f"excited_state_route_is_mainline: {summary.excited_state_route_is_mainline}",
        f"states: {summary.states}",
        f"independent_state_params: {summary.independent_state_params}",
        f"independent_state_merge_keys: {summary.independent_state_merge_keys}",
        f"diagonal_mcmc_trace: {summary.diagonal_mcmc_trace}",
        f"diagonal_local_energy: {summary.diagonal_local_energy}",
        f"diagonal_overlap_jvp: {summary.diagonal_overlap_jvp}",
        f"profile_step_times: {summary.profile_step_times}",
        f"profile_loss_components: {summary.profile_loss_components}",
        f"overlap_penalty: {summary.overlap_penalty}",
        f"overlap_weights: {summary.overlap_weights}",
        f"overlap_scale_by: {summary.overlap_scale_by}",
        f"overlap_min_scale: {summary.overlap_min_scale}",
        f"overlap_max_scale: {summary.overlap_max_scale}",
        f"overlap_clip_width: {summary.overlap_clip_width}",
        f"overlap_clip_exclude_width: {summary.overlap_clip_exclude_width}",
        f"overlap_sort_states_by: {summary.overlap_sort_states_by}",
        f"overlap_use_ewm_scale: {summary.overlap_use_ewm_scale}",
        f"overlap_ewm_max_alpha: {summary.overlap_ewm_max_alpha}",
        f"overlap_ewm_decay_alpha: {summary.overlap_ewm_decay_alpha}",
        f"fixed_ground_checkpoint: {summary.fixed_ground_checkpoint}",
        (
            "fixed_ground_init_trainable_from_checkpoint: "
            f"{summary.fixed_ground_init_trainable_from_checkpoint}"
        ),
        (
            "fixed_ground_init_trainable_noise_scale: "
            f"{summary.fixed_ground_init_trainable_noise_scale}"
        ),
        f"fixed_ground_overlap_penalty: {summary.fixed_ground_overlap_penalty}",
        (
            "fixed_ground_overlap_clip_width: "
            f"{summary.fixed_ground_overlap_clip_width}"
        ),
        (
            "fixed_ground_overlap_clip_exclude_width: "
            f"{summary.fixed_ground_overlap_clip_exclude_width}"
        ),
        (
            "fixed_ground_overlap_gradient_scale: "
            f"{summary.fixed_ground_overlap_gradient_scale}"
        ),
        (
            "fixed_ground_symmetric_sampling: "
            f"{summary.fixed_ground_symmetric_sampling}"
        ),
        f"fixed_ground_energy_reference: {summary.fixed_ground_energy_reference}",
        f"spin_penalty: {summary.spin_penalty}",
        f"log_spin_by_state: {summary.log_spin_by_state}",
        f"s2_observable: {summary.s2_observable}",
        f"check_nan: {summary.check_nan}",
        f"reset_if_nan: {summary.reset_if_nan}",
        f"kfac_norm_constraint: {summary.kfac_norm_constraint}",
        f"kfac_norm_constraint_scale_by_states: {summary.kfac_norm_constraint_scale_by_states}",
        f"iterations: {summary.iterations}",
        f"batch_size: {summary.batch_size}",
        f"laplacian: {summary.laplacian}",
        f"forward_laplacian_enabled: {summary.forward_laplacian_enabled}",
        f"pretrain_method: {summary.pretrain_method}",
        f"pretrain_iterations: {summary.pretrain_iterations}",
        f"pretrain_target_backend: {summary.pretrain_target_backend}",
        f"pretrain_jax_pbc_image_cutoff: {summary.pretrain_jax_pbc_image_cutoff}",
        f"pretrain_log_every: {summary.pretrain_log_every}",
        f"pretrain_target_chunk_size: {summary.pretrain_target_chunk_size}",
        f"mcmc_burn_in: {summary.mcmc_burn_in}",
        f"mcmc_steps_per_iteration: {summary.mcmc_steps_per_iteration}",
        "",
        f"target_jax_version: {summary.target_jax_version}",
        f"precision_profile: {summary.precision_profile}",
        f"x64_enabled: {summary.x64_enabled}",
        f"psiformer_tf32: {summary.psiformer_tf32}",
        (
            "psiformer_attention_implementation: "
            f"{summary.psiformer_attention_implementation}"
        ),
        f"psiformer_attention_kernel_gpu: {summary.psiformer_attention_kernel_gpu}",
        "",
        f"save_path: {summary.save_path}",
        f"restore_path: {summary.restore_path}",
    ]
    return "\n".join(lines)


def _method_profile_defaults(method_profile: str | None) -> dict[str, Any]:
    """Return SolidNES method-profile defaults for FermiNet config expansion."""

    if method_profile in (None, "", "none"):
        return {}
    if method_profile == "szabo_noe_2024_penalty":
        return {
            "overlap_penalty": 4.0,
            "overlap_equal_weights": True,
            "overlap_scale_by": "max_gap_std",
            "overlap_min_scale": 0.001,
            "overlap_max_scale": 5.0,
            "overlap_clip_width": 10.0,
            "overlap_clip_exclude_width": float("inf"),
            "overlap_sort_states_by": None,
            "overlap_use_ewm_scale": True,
            "overlap_ewm_max_alpha": 0.999,
            "overlap_ewm_decay_alpha": 10.0,
            "independent_state_params": True,
            "kfac_norm_constraint_scale_by_states": False,
        }
    raise ValueError(f"Unsupported FermiNet method_profile: {method_profile}")


def _build_ferminet_config(
    experiment: dict[str, Any],
    system: dict[str, Any],
    model: dict[str, Any],
    sampler: dict[str, Any],
    train: dict[str, Any],
    project_root: Path,
) -> Any:
    template = experiment["ferminet_config_template"]
    module = importlib.import_module(module_from_ferminet_template(template["module"]))
    cfg = module.get_config()

    model_cfg = model["model"]
    cfg.network.network_type = str(model_cfg.get("network_type", model_cfg["ansatz"]))
    cfg.network.determinants = int(model_cfg["determinants"])
    cfg.network.full_det = bool(model_cfg.get("full_det", True))
    cfg.network.complex = bool(model_cfg.get("complex", False))
    if cfg.network.network_type == "ferminet":
        cfg.network.ferminet.hidden_dims = _as_tuple_layers(model_cfg["hidden_dims"])
        cfg.network.ferminet.use_last_layer = bool(model_cfg.get("use_last_layer", False))
        cfg.network.ferminet.separate_spin_channels = bool(
            model_cfg.get("separate_spin_channels", False)
        )
    elif cfg.network.network_type == "psiformer":
        psiformer_cfg = model_cfg["psiformer"]
        cfg.network.psiformer.num_layers = int(psiformer_cfg["num_layers"])
        cfg.network.psiformer.num_heads = int(psiformer_cfg["num_heads"])
        cfg.network.psiformer.heads_dim = int(psiformer_cfg["heads_dim"])
        cfg.network.psiformer.mlp_hidden_dims = tuple(
            int(x) for x in psiformer_cfg["mlp_hidden_dims"]
        )
        cfg.network.psiformer.use_layer_norm = bool(psiformer_cfg["use_layer_norm"])
        cfg.network.psiformer.tf32 = bool(psiformer_cfg.get("tf32", True))
        cfg.network.psiformer_attention_implementation = str(
            model_cfg.get("attention", {}).get(
                "implementation",
                psiformer_cfg.get("attention_implementation", "auto"),
            )
        )
        cfg.network.psiformer_attention_kernel_gpu = str(
            model_cfg.get("attention", {}).get(
                "kernel_gpu",
                psiformer_cfg.get("attention_kernel_gpu", "jax"),
            )
        )
    else:
        raise ValueError(f"Unsupported FermiNet network_type: {cfg.network.network_type}")

    sampler_cfg = sampler["sampler"]
    train_cfg = train["training"]
    method_profile = train_cfg.get("method_profile")
    profile_defaults = _method_profile_defaults(method_profile)
    cfg.batch_size = int(train_cfg["batch_size"])
    cfg.debug.deterministic = bool(train_cfg.get("deterministic", True))
    cfg.debug.check_nan = bool(train_cfg.get("check_nan", False))
    cfg.optim.iterations = int(train_cfg["iterations"])
    cfg.optim.optimizer = str(train_cfg["optimizer"])
    cfg.optim.reset_if_nan = bool(
        train_cfg.get("reset_if_nan", cfg.optim.get("reset_if_nan", False))
    )
    cfg.optim.objective = str(train_cfg.get("objective", cfg.optim.objective))
    cfg.optim.method_profile = method_profile
    cfg.system.states = int(
        train_cfg.get(
            "states",
            train_cfg.get("excited_states", cfg.system.get("states", 0)),
        )
    )
    cfg.network.independent_states = bool(
        train_cfg.get(
            "independent_state_params",
            train_cfg.get(
                "independent_states",
                profile_defaults.get("independent_state_params", False),
            ),
        )
    )
    parameter_sharing = dict(train_cfg.get("parameter_sharing", {}))
    cfg.network.independent_state_merge_keys = _as_tuple_strings(
        train_cfg.get(
            "independent_state_merge_keys",
            parameter_sharing.get("merge_keys", ()),
        )
    )
    diagonal_paths = dict(train_cfg.get("diagonal_paths", {}))
    cfg.optim.diagonal_mcmc_trace = _as_optional_bool(
        train_cfg.get("diagonal_mcmc_trace", diagonal_paths.get("mcmc_trace"))
    )
    cfg.optim.diagonal_local_energy = _as_optional_bool(
        train_cfg.get("diagonal_local_energy", diagonal_paths.get("local_energy"))
    )
    cfg.optim.diagonal_overlap_jvp = _as_optional_bool(
        train_cfg.get("diagonal_overlap_jvp", diagonal_paths.get("overlap_jvp"))
    )
    profiling = dict(train_cfg.get("profiling", {}))
    cfg.log.profile_step_times = bool(
        train_cfg.get("profile_step_times", profiling.get("step_times", False))
    )
    cfg.log.profile_loss_components = bool(
        train_cfg.get(
            "profile_loss_components", profiling.get("loss_components", False)
        )
    )
    if cfg.optim.objective == "vmc_overlap" and cfg.system.states <= 0:
        raise ValueError("FermiNet vmc_overlap objective requires training.states > 0")
    cfg.optim.overlap.penalty = float(
        train_cfg.get(
            "overlap_penalty",
            profile_defaults.get(
                "overlap_penalty",
                4.0
                if cfg.optim.objective == "vmc_overlap"
                else cfg.optim.overlap.penalty,
            ),
        )
    )
    if "overlap_weights" in train_cfg:
        overlap_weights = train_cfg.get("overlap_weights")
        if overlap_weights is None and profile_defaults.get("overlap_equal_weights", False):
            cfg.optim.overlap.weights = tuple(
                [1.0 / float(cfg.system.states)] * int(cfg.system.states)
            )
        else:
            cfg.optim.overlap.weights = (
                None
                if overlap_weights is None
                else tuple(float(weight) for weight in overlap_weights)
            )
    elif profile_defaults.get("overlap_equal_weights", False):
        cfg.optim.overlap.weights = tuple(
            [1.0 / float(cfg.system.states)] * int(cfg.system.states)
        )
    cfg.optim.overlap.scale_by = train_cfg.get(
        "overlap_scale_by",
        profile_defaults.get(
            "overlap_scale_by",
            "max_gap_std" if cfg.optim.objective == "vmc_overlap" else None,
        ),
    )
    cfg.optim.overlap.min_scale = float(
        train_cfg.get(
            "overlap_min_scale",
            profile_defaults.get("overlap_min_scale", 0.001),
        )
    )
    cfg.optim.overlap.max_scale = float(
        train_cfg.get(
            "overlap_max_scale",
            profile_defaults.get("overlap_max_scale", 5.0),
        )
    )
    cfg.optim.overlap.clip_width = float(
        train_cfg.get(
            "overlap_clip_width",
            profile_defaults.get("overlap_clip_width", 10.0),
        )
    )
    cfg.optim.overlap.clip_exclude_width = float(
        train_cfg.get(
            "overlap_clip_exclude_width",
            profile_defaults.get("overlap_clip_exclude_width", float("inf")),
        )
    )
    cfg.optim.overlap.sort_states_by = train_cfg.get(
        "overlap_sort_states_by",
        profile_defaults.get("overlap_sort_states_by"),
    )
    cfg.optim.overlap.use_ewm_scale = bool(
        train_cfg.get(
            "overlap_use_ewm_scale",
            profile_defaults.get("overlap_use_ewm_scale", False),
        )
    )
    cfg.optim.overlap.ewm_max_alpha = float(
        train_cfg.get(
            "overlap_ewm_max_alpha",
            profile_defaults.get("overlap_ewm_max_alpha", 0.999),
        )
    )
    cfg.optim.overlap.ewm_decay_alpha = float(
        train_cfg.get(
            "overlap_ewm_decay_alpha",
            profile_defaults.get("overlap_ewm_decay_alpha", 10.0),
        )
    )
    fixed_ground_cfg = dict(train_cfg.get("fixed_ground", {}))
    if cfg.optim.objective == "fixed_ground_overlap":
        if cfg.system.states:
            raise ValueError(
                "FermiNet fixed_ground_overlap trains one excited state at a time; "
                "set training.states to 0."
            )
        checkpoint_value = fixed_ground_cfg.get(
            "checkpoint", train_cfg.get("fixed_ground_checkpoint")
        )
        if checkpoint_value is None:
            raise ValueError(
                "FermiNet fixed_ground_overlap requires fixed_ground.checkpoint"
            )
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        fixed_ground = ml_collections.ConfigDict()
        fixed_ground.checkpoint = str(checkpoint_path)
        fixed_ground.init_trainable_from_checkpoint = bool(
            fixed_ground_cfg.get(
                "init_trainable_from_checkpoint",
                train_cfg.get("fixed_ground_init_trainable_from_checkpoint", True),
            )
        )
        fixed_ground.init_trainable_noise_scale = float(
            fixed_ground_cfg.get(
                "init_trainable_noise_scale",
                train_cfg.get("fixed_ground_init_trainable_noise_scale", 0.0),
            )
        )
        fixed_ground.overlap_penalty = float(
            fixed_ground_cfg.get(
                "overlap_penalty",
                train_cfg.get("overlap_penalty", cfg.optim.overlap.penalty),
            )
        )
        fixed_ground.clip_width = float(
            fixed_ground_cfg.get(
                "clip_width",
                train_cfg.get("overlap_clip_width", cfg.optim.overlap.clip_width),
            )
        )
        fixed_ground.clip_exclude_width = float(
            fixed_ground_cfg.get(
                "clip_exclude_width",
                train_cfg.get(
                    "overlap_clip_exclude_width",
                    cfg.optim.overlap.clip_exclude_width,
                ),
            )
        )
        fixed_ground.gradient_scale = float(
            fixed_ground_cfg.get(
                "gradient_scale",
                train_cfg.get("fixed_ground_overlap_gradient_scale", 1.0),
            )
        )
        fixed_ground.symmetric_sampling = bool(
            fixed_ground_cfg.get(
                "symmetric_sampling",
                train_cfg.get("fixed_ground_symmetric_sampling", True),
            )
        )
        energy_reference = fixed_ground_cfg.get(
            "energy_reference", train_cfg.get("fixed_ground_energy_reference")
        )
        fixed_ground.energy_reference = (
            None if energy_reference is None else float(energy_reference)
        )
        cfg.optim.fixed_ground = fixed_ground
    else:
        cfg.optim.fixed_ground = ml_collections.ConfigDict()
    cfg.optim.laplacian = str(train_cfg.get("laplacian", "folx"))
    cfg.optim.max_vmap_batch_size = int(train_cfg.get("max_vmap_batch_size", 0))
    cfg.optim.max_local_energy_vmap_batch_size = int(
        train_cfg.get(
            "max_local_energy_vmap_batch_size",
            cfg.optim.max_vmap_batch_size,
        )
    )
    cfg.optim.clip_local_energy = float(train_cfg.get("clip_local_energy", 5.0))
    cfg.optim.clip_median = bool(train_cfg.get("clip_median", False))
    cfg.optim.center_at_clip = bool(train_cfg.get("center_at_clip", True))
    spin_penalty = train_cfg.get("spin_penalty", train_cfg.get("spin_energy", 0.0))
    cfg.optim.spin_energy = float(spin_penalty)
    cfg.optim.log_spin_by_state = bool(
        train_cfg.get(
            "log_spin_by_state",
            cfg.optim.spin_energy > 0.0 and int(cfg.system.states) > 0,
        )
    )
    cfg.observables.s2 = bool(
        train_cfg.get(
            "observables_s2",
            train_cfg.get("s2_observable", False),
        )
    )
    cfg.optim.lr.rate = float(train_cfg.get("learning_rate", cfg.optim.lr.rate))
    cfg.optim.lr.decay = float(train_cfg.get("learning_rate_decay", cfg.optim.lr.decay))
    cfg.optim.lr.delay = float(train_cfg.get("learning_rate_delay", cfg.optim.lr.delay))
    norm_constraint_scale_by_states = bool(
        train_cfg.get(
            "kfac_norm_constraint_scale_by_states",
            profile_defaults.get("kfac_norm_constraint_scale_by_states", False),
        )
    )
    if "kfac" in train_cfg:
        kfac_cfg = dict(train_cfg["kfac"])
        norm_constraint_scale_by_states = bool(
            kfac_cfg.pop(
                "norm_constraint_scale_by_states",
                norm_constraint_scale_by_states,
            )
        )
        for key, value in kfac_cfg.items():
            if key not in cfg.optim.kfac:
                raise ValueError(f"Unknown FermiNet KFAC option in train config: {key}")
            cfg.optim.kfac[key] = value
    cfg.optim.kfac.norm_constraint_scale_by_states = norm_constraint_scale_by_states
    if (
        cfg.optim.objective == "vmc_overlap"
        and cfg.optim.optimizer == "kfac"
        and norm_constraint_scale_by_states
        and cfg.system.states > 1
    ):
        cfg.optim.kfac.norm_constraint = (
            float(cfg.optim.kfac.norm_constraint) * int(cfg.system.states)
        )
    cfg.pretrain.method = train_cfg.get("pretrain_method")
    cfg.pretrain.iterations = int(train_cfg.get("pretrain_iterations", 0))
    system_section = system.get("system", {})
    system_pseudopotential = system_section.get("pseudopotential", {})
    cfg.pretrain.basis = str(
        train_cfg.get(
            "pretrain_basis",
            system_section.get(
                "basis",
                system_pseudopotential.get("basis", cfg.pretrain.basis)
                if isinstance(system_pseudopotential, dict)
                else cfg.pretrain.basis,
            ),
        )
    )
    cfg.pretrain.learning_rate = float(
        train_cfg.get("pretrain_learning_rate", cfg.pretrain.get("learning_rate", 3e-4))
    )
    cfg.pretrain.mcmc_steps = int(
        train_cfg.get("pretrain_mcmc_steps", cfg.pretrain.get("mcmc_steps", 1))
    )
    cfg.pretrain.log_every = int(
        train_cfg.get("pretrain_log_every", cfg.pretrain.get("log_every", 1))
    )
    cfg.pretrain.target_chunk_size = int(
        train_cfg.get(
            "pretrain_target_chunk_size",
            cfg.pretrain.get("target_chunk_size", 0),
        )
    )
    cfg.pretrain.target_backend = str(
        train_cfg.get(
            "pretrain_target_backend",
            cfg.pretrain.get("target_backend", "pyscf_pbc"),
        )
    )
    cfg.pretrain.jax_pbc_image_cutoff = int(
        train_cfg.get(
            "pretrain_jax_pbc_image_cutoff",
            cfg.pretrain.get("jax_pbc_image_cutoff", 2),
        )
    )
    cfg.pretrain.pbc = bool(
        train_cfg.get(
            "pretrain_pbc",
            cfg.pretrain.method == "pbc_hf",
        )
    )
    cfg.pretrain.scf_fraction = float(
        train_cfg.get(
            "pretrain_scf_fraction",
            0.0 if cfg.pretrain.pbc else cfg.pretrain.get("scf_fraction", 1.0),
        )
    )
    if cfg.pretrain.pbc:
        if cfg.pretrain.target_backend == "jax_pbc_gto":
            pretrain_basis = cfg.pretrain.basis.lower().replace("_", "-")
            if pretrain_basis not in {"sto-3g", "ccpvdz", "cc-pvdz"}:
                raise ValueError(
                    "pretrain_target_backend=jax_pbc_gto is currently validated "
                    "only for pretrain_basis=sto-3g or ccpvdz."
                )
            if (
                pretrain_basis in {"ccpvdz", "cc-pvdz"}
                and int(cfg.pretrain.jax_pbc_image_cutoff) < 3
            ):
                raise ValueError(
                    "pretrain_target_backend=jax_pbc_gto with ccpvdz requires "
                    "pretrain_jax_pbc_image_cutoff >= 3."
                )
        cfg.pretrain.lattice = cfg.system.make_local_energy_kwargs.get("lattice")
        cfg.pretrain.twist = tuple(
            float(x) for x in system_section.get("twist", [0.0, 0.0, 0.0])
        )
        cfg.pretrain.cell_precision = float(train_cfg.get("pretrain_precision", 1e-8))
        cfg.pretrain.cell_exp_to_discard = float(
            train_cfg.get("pretrain_exp_to_discard", 0.1)
        )
        cfg.pretrain.restricted = bool(train_cfg.get("pretrain_restricted", True))
        pseudo = train_cfg.get("pretrain_pseudo") or system_section.get("pseudo")
        if not pseudo and isinstance(system_pseudopotential, dict):
            pseudo = system_pseudopotential.get("pseudo")
        if pseudo:
            cfg.pretrain.pseudo = pseudo
        if any(abs(x) > 1e-12 for x in cfg.pretrain.twist) and not cfg.network.get(
            "complex", False
        ):
            raise ValueError(
                "PBC HF pretraining with nonzero twist requires model complex: true."
            )
    cfg.mcmc.burn_in = int(sampler_cfg["burn_in"])
    cfg.mcmc.steps = int(sampler_cfg["steps_per_iteration"])
    cfg.mcmc.move_width = float(sampler_cfg["proposal_width"])
    cfg.mcmc.adapt_frequency = int(sampler_cfg["adapt_frequency"])
    cfg.mcmc.blocks = int(sampler_cfg.get("blocks", 1))
    cfg.log.stats_frequency = int(train_cfg.get("log_every", 1))
    cfg.log.save_frequency = float(train_cfg.get("checkpoint_every_minutes", 30))

    output = experiment["output"]
    cfg.log.save_path = str((project_root / output["checkpoint_dir"]).resolve())
    if "restore_checkpoint_dir" in output:
        restore_path = (project_root / output["restore_checkpoint_dir"]).resolve()
        if not restore_path.exists():
            raise FileNotFoundError(restore_path)
        cfg.log.restore_path = str(restore_path)
    return cfg


def module_from_ferminet_template(template_module: str) -> str:
    if template_module.endswith(".py"):
        template_module = template_module[:-3]
    return template_module.replace("/", ".")


def _as_tuple_layers(hidden_dims: list[list[int]]) -> tuple[tuple[int, int], ...]:
    layers = []
    for layer in hidden_dims:
        if len(layer) != 2:
            raise ValueError(f"Expected [one_body, two_body] layer, got {layer}")
        layers.append((int(layer[0]), int(layer[1])))
    return tuple(layers)


def _as_tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _resolve_auto_bool(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _ensure_import_paths(project_root: Path, ferminet_root: Path) -> None:
    for path in (project_root / "src", ferminet_root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
