"""Adapter helpers for building DeepSolid configs from SolidNES YAML."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib
import io
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DeepSolidAdapterPaths:
    """Resolved config paths used to build a DeepSolid config."""

    experiment: Path
    system: Path
    model: Path
    sampler: Path
    train: Path


@dataclass(frozen=True)
class DeepSolidRuntimeCompatibility:
    """Runtime facts that matter before launching DeepSolid."""

    use_x64: bool
    optimizer: str
    iterations: int
    compatibility_shims_required: bool
    kfac_neutralized_for_smoke: bool


@dataclass(frozen=True)
class DeepSolidAdapterSummary:
    """Small serializable summary of the adapter output."""

    experiment_name: str
    backend: str
    template_module: str
    input_string: str
    nelectron: int
    nelec: Any
    dimension: int
    basis: Any
    pseudo: Any
    ecp_or_pseudopotential: Any
    batch_size: int
    rng_seed: int | None
    params_seed: int | None
    use_x64: bool
    optimizer: str
    iterations: int
    pretrain_iterations: int
    mcmc_burn_in: int
    mcmc_steps_per_iteration: int
    hidden_dims: Any
    determinants: int
    distance_type: str
    envelope_type: str
    save_path: str
    restore_path: str
    runtime_compatibility: DeepSolidRuntimeCompatibility

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "backend": self.backend,
            "template_module": self.template_module,
            "input_string": self.input_string,
            "nelectron": self.nelectron,
            "nelec": self.nelec,
            "dimension": self.dimension,
            "basis": self.basis,
            "pseudo": self.pseudo,
            "ecp_or_pseudopotential": self.ecp_or_pseudopotential,
            "batch_size": self.batch_size,
            "rng_seed": self.rng_seed,
            "params_seed": self.params_seed,
            "use_x64": self.use_x64,
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "pretrain_iterations": self.pretrain_iterations,
            "mcmc_burn_in": self.mcmc_burn_in,
            "mcmc_steps_per_iteration": self.mcmc_steps_per_iteration,
            "hidden_dims": self.hidden_dims,
            "determinants": self.determinants,
            "distance_type": self.distance_type,
            "envelope_type": self.envelope_type,
            "save_path": self.save_path,
            "restore_path": self.restore_path,
            "runtime_compatibility": {
                "use_x64": self.runtime_compatibility.use_x64,
                "optimizer": self.runtime_compatibility.optimizer,
                "iterations": self.runtime_compatibility.iterations,
                "compatibility_shims_required": self.runtime_compatibility.compatibility_shims_required,
                "kfac_neutralized_for_smoke": self.runtime_compatibility.kfac_neutralized_for_smoke,
            },
        }


@dataclass(frozen=True)
class DeepSolidAdapterBundle:
    """Built DeepSolid config plus the SolidNES inputs that produced it."""

    cfg: Any
    experiment: dict[str, Any]
    system: dict[str, Any]
    model: dict[str, Any]
    sampler: dict[str, Any]
    train: dict[str, Any]
    paths: DeepSolidAdapterPaths

    @property
    def summary(self) -> DeepSolidAdapterSummary:
        cell = self.cfg.system.pyscf_cell
        system_notes = self.system.get("notes", {})
        return DeepSolidAdapterSummary(
            experiment_name=self.experiment["experiment_name"],
            backend=self.experiment["backend"]["name"],
            template_module=self.experiment["deepsolid_config_template"]["module"],
            input_string=self.experiment["deepsolid_config_template"]["input_string"],
            nelectron=cell.nelectron,
            nelec=cell.nelec,
            dimension=cell.dimension,
            basis=cell.basis,
            pseudo=cell.pseudo,
            ecp_or_pseudopotential=_ecp_status(self.system),
            batch_size=self.cfg.batch_size,
            rng_seed=getattr(self.cfg.debug, "seed", None),
            params_seed=getattr(self.cfg.debug, "params_seed", None),
            use_x64=self.cfg.use_x64,
            optimizer=self.cfg.optim.optimizer,
            iterations=self.cfg.optim.iterations,
            pretrain_iterations=self.cfg.pretrain.iterations,
            mcmc_burn_in=self.cfg.mcmc.burn_in,
            mcmc_steps_per_iteration=self.cfg.mcmc.steps,
            hidden_dims=self.cfg.network.detnet.hidden_dims,
            determinants=self.cfg.network.detnet.determinants,
            distance_type=self.cfg.network.detnet.distance_type,
            envelope_type=self.cfg.network.detnet.envelope_type,
            save_path=self.cfg.log.save_path,
            restore_path=self.cfg.log.restore_path,
            runtime_compatibility=DeepSolidRuntimeCompatibility(
                use_x64=self.cfg.use_x64,
                optimizer=self.cfg.optim.optimizer,
                iterations=self.cfg.optim.iterations,
                compatibility_shims_required=True,
                kfac_neutralized_for_smoke=self.cfg.optim.optimizer != "kfac",
            ),
        )


@dataclass
class DeepSolidGroundStateObjects:
    """Initialized DeepSolid runtime pieces needed by SolidNES."""

    cfg: Any
    simulation_cell: Any
    hartree_fock: Any
    slater_mat: Any
    slater_logdet: Any
    slater_slogdet: Any
    batch_slater_mat: Any
    batch_slater_logdet: Any
    batch_slater_slogdet: Any
    params: Any
    data: Any
    total_energy: Any
    mcmc_step: Any
    mcmc_width: Any
    sharded_key: Any
    local_batch_size: int
    num_devices: int

    def evaluate_energy(self) -> Any:
        """Evaluate the pmap-wrapped DeepSolid local-energy loss."""

        from DeepSolid import constants  # pylint: disable=import-outside-toplevel

        return constants.pmap(self.total_energy)(self.params, self.data)

    def run_mcmc_step(self) -> tuple[Any, Any]:
        """Run one pmap-wrapped MCMC step and update the stored walkers."""

        from DeepSolid import constants  # pylint: disable=import-outside-toplevel

        self.sharded_key, subkeys = constants.p_split(self.sharded_key)
        self.data, pmove = self.mcmc_step(self.params, self.data, subkeys, self.mcmc_width)
        return self.data, pmove


@contextlib.contextmanager
def suppress_native_output():
    """Suppress noisy native/PySCF output during config construction."""

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def build_deepsolid_adapter(
    experiment_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    verbose_pyscf: bool | None = None,
) -> DeepSolidAdapterBundle:
    """Build a DeepSolid config and retain SolidNES-side metadata."""

    resolved_experiment_path = _resolve_project_path(experiment_path, project_root)
    experiment = load_yaml(resolved_experiment_path)
    paths = DeepSolidAdapterPaths(
        experiment=resolved_experiment_path,
        system=resolve_config_path(resolved_experiment_path, experiment["system_config"]),
        model=resolve_config_path(resolved_experiment_path, experiment["model_config"]),
        sampler=resolve_config_path(resolved_experiment_path, experiment["sampler_config"]),
        train=resolve_config_path(resolved_experiment_path, experiment["train_config"]),
    )

    system = load_yaml(paths.system)
    model = load_yaml(paths.model)
    sampler = load_yaml(paths.sampler)
    train = load_yaml(paths.train)
    cfg = _build_deepsolid_config(experiment, model, sampler, train, project_root, verbose_pyscf)
    return DeepSolidAdapterBundle(cfg, experiment, system, model, sampler, train, paths)


def create_output_dirs(bundle: DeepSolidAdapterBundle) -> None:
    Path(bundle.cfg.log.save_path).mkdir(parents=True, exist_ok=True)
    output = bundle.experiment.get("output", {})
    log_dir = output.get("log_dir")
    if log_dir:
        Path(PROJECT_ROOT / log_dir).mkdir(parents=True, exist_ok=True)


def initialize_deepsolid_ground_state(
    bundle: DeepSolidAdapterBundle,
    *,
    apply_smoke_shims: bool = True,
    verbose_hf: bool | None = None,
) -> DeepSolidGroundStateObjects:
    """Initialize DeepSolid internals without entering its training loop."""

    if apply_smoke_shims:
        from solidnes.backends.deepsolid_compat import apply_jax_legacy_shims
        from solidnes.backends.deepsolid_compat import neutralize_kfac_tags_for_smoke

        apply_jax_legacy_shims()
        neutralize_kfac_tags_for_smoke()

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from DeepSolid import constants  # pylint: disable=import-outside-toplevel
    from DeepSolid import hf  # pylint: disable=import-outside-toplevel
    from DeepSolid import init_guess  # pylint: disable=import-outside-toplevel
    from DeepSolid import network  # pylint: disable=import-outside-toplevel
    from DeepSolid import qmc  # pylint: disable=import-outside-toplevel
    from DeepSolid import train  # pylint: disable=import-outside-toplevel

    cfg = bundle.cfg
    if cfg.use_x64:
        jax.config.update("jax_enable_x64", True)

    num_devices = jax.local_device_count()
    local_batch_size = cfg.batch_size
    if local_batch_size % num_devices != 0:
        raise ValueError(
            "Batch size must be divisible by number of devices, "
            f"got batch size {local_batch_size} for {num_devices} devices."
        )
    if cfg.system.ndim != 3:
        raise ValueError("Only 3D systems are currently supported.")

    simulation_cell = cfg.system.pyscf_cell
    cfg.system.internal_cell = init_guess.pyscf_to_cell(cell=simulation_cell)

    hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
    show_hf_output = os.environ.get("SOLIDNES_VERBOSE_PYSCF") == "1" if verbose_hf is None else verbose_hf
    if show_hf_output:
        hartree_fock.init_scf()
    else:
        with suppress_native_output():
            hartree_fock.init_scf()

    data_shape = (num_devices, local_batch_size // num_devices)
    seed = getattr(cfg.debug, "seed", None)
    if seed is None:
        seed = 666 if cfg.debug.deterministic else int(1e6 * time.time())
    else:
        seed = int(seed)
    key = jax.random.PRNGKey(seed)

    system_dict = {
        "klist": hartree_fock.klist,
        "simulation_cell": simulation_cell,
    }
    system_dict.update(cfg.network.detnet)

    slater_mat = network.make_solid_fermi_net(**system_dict, method_name="eval_mats")
    slater_logdet = network.make_solid_fermi_net(**system_dict, method_name="eval_logdet")
    slater_slogdet = network.make_solid_fermi_net(**system_dict, method_name="eval_slogdet")

    batch_slater_mat = jax.vmap(slater_mat.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0), out_axes=0)

    data = init_guess.init_electrons(
        key=key,
        cell=cfg.system.internal_cell,
        latvec=simulation_cell.lattice_vectors(),
        electrons=simulation_cell.nelec,
        batch_size=local_batch_size,
        init_width=cfg.mcmc.init_width,
    )
    data = jnp.reshape(data, data_shape + data.shape[1:])
    data = constants.broadcast_all_local_devices(data)

    params_seed = getattr(cfg.debug, "params_seed", None)
    if params_seed is None:
        if getattr(cfg.debug, "seed", None) is not None:
            params_seed = int(cfg.debug.seed) + 222
        else:
            params_seed = 888 if cfg.debug.deterministic else int(1e6 * time.time())
    else:
        params_seed = int(params_seed)
    params_key = jax.random.PRNGKey(params_seed)
    params = slater_logdet.init(key=params_key, data=None)
    params = constants.replicate_all_local_devices(params)

    sampling_func = slater_slogdet.apply if cfg.mcmc.importance_sampling else None
    raw_mcmc_step = qmc.make_mcmc_step(
        batch_slog_network=batch_slater_slogdet,
        batch_per_device=local_batch_size // num_devices,
        latvec=jnp.asarray(simulation_cell.lattice_vectors()),
        steps=cfg.mcmc.steps,
        one_electron_moves=cfg.mcmc.one_electron,
        importance_sampling=sampling_func,
    )
    total_energy = train.make_loss(
        network=slater_logdet.apply,
        batch_network=batch_slater_logdet,
        simulation_cell=simulation_cell,
        clip_local_energy=cfg.optim.clip_el,
        clip_type=cfg.optim.clip_type,
        mode=cfg.optim.laplacian_mode,
        partition_number=cfg.optim.partition_number,
    )

    return DeepSolidGroundStateObjects(
        cfg=cfg,
        simulation_cell=simulation_cell,
        hartree_fock=hartree_fock,
        slater_mat=slater_mat,
        slater_logdet=slater_logdet,
        slater_slogdet=slater_slogdet,
        batch_slater_mat=batch_slater_mat,
        batch_slater_logdet=batch_slater_logdet,
        batch_slater_slogdet=batch_slater_slogdet,
        params=params,
        data=data,
        total_energy=total_energy,
        mcmc_step=constants.pmap(raw_mcmc_step),
        mcmc_width=constants.replicate_all_local_devices(jnp.asarray(cfg.mcmc.move_width)),
        sharded_key=constants.make_different_rng_key_on_all_devices(key),
        local_batch_size=local_batch_size,
        num_devices=num_devices,
    )


def format_summary(summary: DeepSolidAdapterSummary) -> str:
    lines = [
        f"experiment: {summary.experiment_name}",
        f"backend: {summary.backend}",
        f"template: {summary.template_module}",
        f"input_string: {summary.input_string}",
        "",
        f"nelectron: {summary.nelectron}",
        f"nelec: {summary.nelec}",
        f"dimension: {summary.dimension}",
        f"basis: {summary.basis}",
        f"pseudo: {summary.pseudo}",
        f"ecp_or_pseudopotential: {summary.ecp_or_pseudopotential}",
        "",
        f"batch_size: {summary.batch_size}",
        f"rng_seed: {summary.rng_seed}",
        f"params_seed: {summary.params_seed}",
        f"use_x64: {summary.use_x64}",
        f"optimizer: {summary.optimizer}",
        f"iterations: {summary.iterations}",
        f"pretrain_iterations: {summary.pretrain_iterations}",
        f"mcmc_burn_in: {summary.mcmc_burn_in}",
        f"mcmc_steps_per_iteration: {summary.mcmc_steps_per_iteration}",
        "",
        f"hidden_dims: {summary.hidden_dims}",
        f"determinants: {summary.determinants}",
        f"distance_type: {summary.distance_type}",
        f"envelope_type: {summary.envelope_type}",
        "",
        f"save_path: {summary.save_path}",
        f"restore_path: {summary.restore_path}",
        "runtime_compatibility:",
        f"  compatibility_shims_required: {summary.runtime_compatibility.compatibility_shims_required}",
        f"  kfac_neutralized_for_smoke: {summary.runtime_compatibility.kfac_neutralized_for_smoke}",
    ]
    return "\n".join(lines)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_config_path(experiment_path: Path, relative_path: str) -> Path:
    path = (experiment_path.parent / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def module_from_deepsolid_template(template_module: str) -> str:
    if template_module.endswith(".py"):
        template_module = template_module[:-3]
    return template_module.replace("/", ".")


def _build_deepsolid_config(
    experiment: dict[str, Any],
    model: dict[str, Any],
    sampler: dict[str, Any],
    train: dict[str, Any],
    project_root: Path,
    verbose_pyscf: bool | None,
) -> Any:
    template = experiment["deepsolid_config_template"]
    module_name = module_from_deepsolid_template(template["module"])
    module = importlib.import_module(module_name)
    show_native_output = os.environ.get("SOLIDNES_VERBOSE_PYSCF") == "1" if verbose_pyscf is None else verbose_pyscf

    if show_native_output:
        cfg = module.get_config(template["input_string"])
    else:
        with suppress_native_output():
            cfg = module.get_config(template["input_string"])

    model_cfg = model["model"]
    cfg.network.detnet.hidden_dims = _as_tuple_layers(model_cfg["hidden_dims"])
    cfg.network.detnet.determinants = int(model_cfg["determinants"])
    cfg.network.detnet.after_determinants = int(model_cfg["after_determinants"])
    cfg.network.detnet.distance_type = model_cfg["distance_type"]
    cfg.network.detnet.envelope_type = model_cfg["envelope_type"]
    cfg.network.detnet.full_det = bool(model_cfg["full_det"])

    sampler_cfg = sampler["sampler"]
    train_cfg = train["training"]
    cfg.batch_size = int(train_cfg["batch_size"])
    if "seed" in train_cfg:
        cfg.debug.seed = int(train_cfg["seed"])
        cfg.debug.params_seed = int(train_cfg.get("params_seed", int(train_cfg["seed"]) + 222))
    cfg.use_x64 = bool(train_cfg["use_x64"])
    cfg.debug.deterministic = bool(train_cfg["deterministic"])
    cfg.optim.iterations = int(train_cfg["iterations"])
    cfg.optim.optimizer = train_cfg["optimizer"]
    cfg.optim.laplacian_mode = train_cfg["laplacian_mode"]
    cfg.optim.partition_number = int(train_cfg["partition_number"])
    if "learning_rate" in train_cfg:
        cfg.optim.lr.rate = float(train_cfg["learning_rate"])
    if "learning_rate_decay" in train_cfg:
        cfg.optim.lr.decay = float(train_cfg["learning_rate_decay"])
    if "learning_rate_delay" in train_cfg:
        cfg.optim.lr.delay = float(train_cfg["learning_rate_delay"])
    if "clip_local_energy" in train_cfg:
        cfg.optim.clip_el = float(train_cfg["clip_local_energy"])
    if "ministeps" in train_cfg:
        cfg.optim.ministeps = int(train_cfg["ministeps"])
    if "kfac" in train_cfg:
        for key, value in train_cfg["kfac"].items():
            if not hasattr(cfg.optim.kfac, key):
                raise ValueError(f"Unknown KFAC option in train config: {key}")
            setattr(cfg.optim.kfac, key, value)
    cfg.pretrain.iterations = int(train_cfg["pretrain_iterations"])
    if "pretrain_method" in train_cfg:
        cfg.pretrain.method = train_cfg["pretrain_method"]
    if "pretrain_learning_rate" in train_cfg:
        cfg.pretrain.lr = float(train_cfg["pretrain_learning_rate"])
    if "pretrain_steps" in train_cfg:
        cfg.pretrain.steps = int(train_cfg["pretrain_steps"])
    cfg.mcmc.burn_in = int(sampler_cfg["burn_in"])
    cfg.mcmc.steps = int(sampler_cfg["steps_per_iteration"])
    cfg.mcmc.move_width = float(sampler_cfg["proposal_width"])
    cfg.mcmc.adapt_frequency = int(sampler_cfg["adapt_frequency"])
    cfg.mcmc.one_electron = bool(sampler_cfg["one_electron"])
    cfg.mcmc.importance_sampling = bool(sampler_cfg["importance_sampling"])
    cfg.log.stats_frequency = int(train_cfg["log_every"])
    if "checkpoint_every_minutes" in train_cfg:
        cfg.log.save_frequency = float(train_cfg["checkpoint_every_minutes"])
    if "checkpoint_every_steps" in train_cfg:
        cfg.log.save_frequency_in_step = int(train_cfg["checkpoint_every_steps"])
    elif int(cfg.optim.iterations) >= 1000:
        # DeepSolid supports step-based checkpointing directly. For long
        # SolidNES jobs, save the final training iteration even if the
        # wall-clock checkpoint timer has not fired.
        cfg.log.save_frequency_in_step = max(1, int(cfg.optim.iterations) - 1)

    output = experiment["output"]
    cfg.log.save_path = str((project_root / output["checkpoint_dir"]).resolve())
    if "restore_checkpoint_dir" in output:
        restore_path = (project_root / output["restore_checkpoint_dir"]).resolve()
        if not restore_path.exists():
            raise FileNotFoundError(restore_path)
        cfg.log.restore_path = str(restore_path)
    cfg.log.stats_file_name = "train_stats"
    return cfg


def _as_tuple_layers(hidden_dims: list[list[int]]) -> tuple[tuple[int, int], ...]:
    layers = []
    for layer in hidden_dims:
        if len(layer) != 2:
            raise ValueError(f"Expected [one_body, two_body] layer, got {layer}")
        layers.append((int(layer[0]), int(layer[1])))
    return tuple(layers)


def _ecp_status(system: dict[str, Any]) -> Any:
    if "pseudopotential" in system:
        return system["pseudopotential"]
    if "ecp_or_pseudopotential" in system:
        return system["ecp_or_pseudopotential"]
    notes = system.get("notes", {})
    if isinstance(notes, dict):
        return notes.get("ecp_or_pseudopotential", notes.get("electron_count_caveat", "not_specified"))
    return "not_specified"


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
