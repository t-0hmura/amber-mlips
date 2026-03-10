# Changelog

## v1.1.0 — 2026-03-11

### New Features

- **ML-Only MD** (`qmmask='@*'`): Full-system MLIP molecular dynamics (non-periodic only)
  - Implicit solvent: xTB ALPB/CPCMX correction via `--solvent <name>` in `ml_keywords`

### Bug Fixes

- Fixed socket hash collision (`id(time)` → `time.time()`)
- Added NaN/Inf validation for cell parameters in C shim
- Added socket path length check before connection

### Other

- New example: `uma_mlonly_implicit.in`
- New documentation: `ML_ONLY_MD.md`
- Binary protocol: fixed response array order (`_RESPONSE_ARRAY_ORDER`)
