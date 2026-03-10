# Changelog

## v1.1.0 — 2026-03-11

### New Features

- **ML-Only MD** (`qmmask='@*'`): Full-system MLIP molecular dynamics
  - PBC: NVT (`ntb=1`) and NPT (`ntb=2`, per-step cell update, `ntwr=1, ntxo=1` required)
  - Implicit solvent: xTB ALPB/CPCMX correction via `--solvent <name>` in `ml_keywords` without PBC

### Bug Fixes

- Fixed socket hash collision (`id(time)` → `time.time()`)
- Added NaN/Inf validation for cell parameters in C shim
- Added socket path length check before connection

### Other

- New examples: `uma_mlonly.in`, `uma_mlonly_npt.in`, `uma_mlonly_implicit.in`
- New documentation: `ML_ONLY_MD.md`
- Binary protocol: fixed response array order (`_RESPONSE_ARRAY_ORDER`)
- PBC + implicit solvent mutual exclusion warning
