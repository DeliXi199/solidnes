# Source

Python source code lives under `src/solidnes/`.

Current source areas:

- `backends/`: adapter code and compatibility helpers for external backends.
- `excited_states/`: backend-independent overlap and penalty-objective helpers
  for the Szabo-Noe excited-state VMC route, the minimal FermiNet PBC
  two-state scaffold, and reusable FermiNet/JAX PBC adapter wrappers for
  externally managed state parameter trees. The adapter source now includes a
  unified FermiNet PBC penalty-term entry point for state energies, overlap
  diagnostics, and penalty objectives.

Keep command-line glue in `scripts/`; keep reusable logic here.
