# Source

Python source code lives under `src/solidnes/`.

Current source areas:

- `backends/`: adapter code and compatibility helpers for external backends.
- `excited_states/`: backend-independent overlap and penalty-objective helpers
  for the Szabo-Noe excited-state VMC route, plus the minimal FermiNet PBC
  two-state scaffold.

Keep command-line glue in `scripts/`; keep reusable logic here.
