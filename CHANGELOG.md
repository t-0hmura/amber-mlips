# Changelog

## v1.0.2 — 2026-03-11

### New Features

- **ML-Only MD** (`qmmask='@*'`): Full-system MLIP molecular dynamics where all atoms are computed by the ML model with no MM force field. AMBER still requires a parm7 topology, but all forces come from the MLIP backend.

- **PBC support (NVT/NPT)**: Cell dimensions are automatically read from the AMBER coordinate file (`-c`) and passed to the MLIP backend.
  - **NVT** (`ntb=1, ntp=0`): Cell is constant throughout the simulation.
  - **NPT** (`ntb=2, ntp=1`): Cell is updated every step by reading the AMBER restart file. Requires `ntwr=1` and `ntxo=1`.
  - Use `--uma-task omat` for PBC-aware UMA models.

- **Implicit solvent correction**: xTB-based ALPB/CPCMX implicit solvation for non-periodic ML-only systems via `--solvent <name>` in `ml_keywords`. Computes `dE = E_xTB(solv) - E_xTB(vac)` and adds the correction to MLIP energy and forces.

- **New examples**: `uma_mlonly.in` (NVT), `uma_mlonly_npt.in` (NPT), `uma_mlonly_implicit.in` (implicit solvent).

- **New documentation**: [`ML_ONLY_MD.md`](ML_ONLY_MD.md) — dedicated guide for ML-only MD setups.

### Improvements

- **Binary protocol hardening**: Response array order is now explicitly fixed via `_RESPONSE_ARRAY_ORDER` in the server, preventing potential data corruption if hessian arrays are added in the future.

- **PBC + implicit solvent warning**: Server now warns at startup if both PBC cell and implicit solvent are active (physically inconsistent configuration).

- **Socket path uniqueness**: Fixed potential socket hash collision when PIDs are reused across sequential runs (`id(time)` → `time.time()`).

- **Cell validation**: C shim now rejects NaN/Inf values when parsing box parameters from AMBER restart files.

- **Socket path length check**: C shim now validates Unix socket path length before attempting connection, providing a clear error message instead of silent truncation.

### ml_keywords Options Added

| Option | Description |
|--------|-------------|
| `--solvent <name>` | xTB implicit solvent (e.g., `water`, `methanol`, `dmso`) |
| `--solvent-model <alpb\|cpcmx>` | Implicit solvent model (default: `alpb`) |

### NPT Requirements

For constant-pressure ML-only MD, two additional AMBER settings are **required**:

| Setting | Value | Reason |
|---------|-------|--------|
| `ntwr` | `1` | Write restart every step for cell updates |
| `ntxo` | `1` | ASCII restart format (C shim reads the box line) |
