"""Adapter helpers for building FermiNet configs from SolidNES YAML."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

from solidnes.backends.deepsolid_adapter import load_yaml
from solidnes.backends.deepsolid_adapter import resolve_config_path


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
        "",
        f"save_path: {summary.save_path}",
        f"restore_path: {summary.restore_path}",
    ]
    return "\n".join(lines)


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
    else:
        raise ValueError(f"Unsupported FermiNet network_type: {cfg.network.network_type}")

    sampler_cfg = sampler["sampler"]
    train_cfg = train["training"]
    cfg.batch_size = int(train_cfg["batch_size"])
    cfg.debug.deterministic = bool(train_cfg.get("deterministic", True))
    cfg.debug.check_nan = bool(train_cfg.get("check_nan", False))
    cfg.optim.iterations = int(train_cfg["iterations"])
    cfg.optim.optimizer = str(train_cfg["optimizer"])
    cfg.optim.laplacian = str(train_cfg.get("laplacian", "folx"))
    cfg.optim.max_vmap_batch_size = int(train_cfg.get("max_vmap_batch_size", 0))
    cfg.optim.clip_local_energy = float(train_cfg.get("clip_local_energy", 5.0))
    cfg.optim.clip_median = bool(train_cfg.get("clip_median", False))
    cfg.optim.center_at_clip = bool(train_cfg.get("center_at_clip", True))
    cfg.optim.lr.rate = float(train_cfg.get("learning_rate", cfg.optim.lr.rate))
    cfg.optim.lr.decay = float(train_cfg.get("learning_rate_decay", cfg.optim.lr.decay))
    cfg.optim.lr.delay = float(train_cfg.get("learning_rate_delay", cfg.optim.lr.delay))
    if "kfac" in train_cfg:
        for key, value in train_cfg["kfac"].items():
            if key not in cfg.optim.kfac:
                raise ValueError(f"Unknown FermiNet KFAC option in train config: {key}")
            cfg.optim.kfac[key] = value
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
