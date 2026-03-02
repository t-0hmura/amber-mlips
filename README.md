# amber-mlips

`amber-mlips` is a single-command wrapper for **AMBER QM/MM** with MLIP backends.
It keeps a `sander`-style CLI and internally uses AMBER `qm_theory='EXTERN'` with `&qc`
through a private non-MPI `qchem` shim.

Supported backend selectors (`&qmmm: qm_theory`):
- `uma`
- `orb`
- `mace`
- `aimnet2`

## Features

- Single command: `amber-mlips ...` (no external server command required).
- Input style remains AMBER-native except two plugin keys:
  - `qm_theory="uma|orb|mace|aimnet2"`
  - `ml_keywords="..."`
- MM side can still use MPI ranks via `--mm-ranks`.

## Installation

Install core package:
```bash
pip install amber-mlips
```

Install with backend extras:
```bash
pip install "amber-mlips[uma]"
pip install "amber-mlips[orb]"
pip install "amber-mlips[mace]"
pip install "amber-mlips[aimnet2]"
```

Local editable install:
```bash
cd amber-mlips
pip install -e . --no-deps
```

## Quick Start

1. Prepare AMBER input (`mlmm.in`):
```text
&qmmm
  qmmask="@1,2,3",
  qmcharge=0,
  spin=1,
  qm_theory="uma",
  ml_keywords="--model uma-s-1p1 --embedcharge",
  qmcut=12.0,
/
```

2. Run with standard `sander` flags:
```bash
amber-mlips -O \
  -i mlmm.in \
  -o mlmm.out \
  -p leap.parm7 \
  -c md.rst7 \
  -r mlmm.rst7 \
  -ref leap.parm7 \
  -x mlmm.nc \
  -inf mlmm.info
```

3. MM MPI parallelism (CPU side):
```bash
amber-mlips --mm-ranks 16 -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 -r mlmm.rst7
```

## Input Semantics

In `&qmmm`, only these keys are plugin-specific:
- `qm_theory` (`uma|orb|mace|aimnet2`)
- `ml_keywords` (backend options string)

Everything else follows native AMBER behavior.
Use `qmcut` directly; no `mlcut` alias is used.

## Internal Transform

Before launching AMBER, `amber-mlips` transforms input as follows:
- `qm_theory` -> `'EXTERN'`
- remove `ml_keywords`
- force `qm_ewald=0` and `qmgb=0` (EXTERN constraints)
- append generated `&qc`

Inspect transformed input with:
```bash
amber-mlips --keep-transformed-input ...
```

## `ml_keywords`

`ml_keywords` is parsed like shell tokens and passed to the internal qchem shim.
Common options:
- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--embedcharge`
- `--xtb-cmd <path_or_cmd>`
- `--xtb-ncores <int>`
- `--debug`

Example:
```text
ml_keywords="--model uma-s-1p1 --embedcharge --xtb-cmd xtb --xtb-ncores 4"
```

## Launcher Behavior

- ML path is non-MPI by design.
- MM side launch is controlled by `--mm-ranks` only.
- Launch policy is fixed to auto:
  - `--mm-ranks 1`: direct `sander`
  - `--mm-ranks > 1`: MPI launcher + `sander.MPI`
- Optional override for launcher binary: `--mpi-bin`.

## Smoke-Test Models

For lightweight integration tests:
- `--model builtin:zero`
- `--model builtin:harmonic`
- `--model builtin:harmonic:<k>`

## Notes

- UMA models may require Hugging Face authentication (`huggingface-cli login`).
- `--embedcharge` requires `xtb` installed and reachable.

## More Docs

- Wrapper and keyword options: [`OPTIONS.md`](OPTIONS.md)
- Implementation details: [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md)
