# Source

Python source code lives under `src/solidnes/`.

Current source areas:

- `backends/`: adapter code and compatibility helpers for external backends.
- `excited_states/`: backend-independent overlap and penalty-objective helpers
  for the Szabo-Noe excited-state VMC route, the minimal FermiNet PBC
  two-state scaffold, and reusable FermiNet/JAX PBC adapter wrappers for
  externally managed state parameter trees. The adapter source now includes a
  unified FermiNet PBC penalty-term entry point for state energies, overlap
  diagnostics, penalty objectives, gradients, a minimal external-state SGD
  update helper, and a fixed-sample training-loop helper that returns per-step
  penalty, state-energy, overlap, and collapse diagnostics.

Keep command-line glue in `scripts/`; keep reusable logic here.
