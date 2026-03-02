# amber-mlips Options

## Wrapper Options (`amber-mlips`)

Wrapper options are consumed by `amber-mlips`; all unrecognized flags are forwarded to `sander`.

- `--sander-bin <path>`: explicit AMBER executable.
- `--mpi-bin <path_or_cmd>`: MPI launcher for MM ranks > 1 (default auto-detect `mpirun`/`mpiexec`).
- `--mm-ranks <int>`: MM-side rank count.
- `--keep-transformed-input`: save transformed mdin as `<input>.amber_mlips.qc.in`.
- `--dry-run`: print transformed command and exit.
- `--debug`: verbose wrapper/shim logs.
- `-h`, `--help`: print help.

Typical use:
```bash
amber-mlips [wrapper-options] -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 ...
```

MM launch behavior is fixed to auto:
- `--mm-ranks 1` => direct `sander`
- `--mm-ranks > 1` => MPI launcher + `sander.MPI`

## User `&qmmm` Fields

Plugin-specific fields:
- `qm_theory="uma"|"orb"|"mace"|"aimnet2"` (required)
- `ml_keywords="..."` (optional, usually required)

On transform:
- `qm_theory` is rewritten to `'EXTERN'`
- `ml_keywords` is removed before AMBER run
- `qm_ewald` and `qmgb` are forced to `0`
- generated `&qc` is appended

All other `&qmmm` fields remain AMBER-native (`qmcut` etc.).

## `ml_keywords` Options

`ml_keywords` is parsed with shell token rules (`shlex.split`) and consumed by the qchem shim.

Common options:
- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--embedcharge`
- `--xtb-cmd <path_or_cmd>`
- `--xtb-acc <float>` (default: `0.2`)
- `--xtb-workdir <tmp|path>` (default: `tmp`)
- `--xtb-keep-files`
- `--xtb-ncores <int>` (default: `1`)
- `--debug`

Backend-specific options:

### UMA
- `--uma-task <omol|omat|odac|oc20|oc25|omc>`
- `--uma-workers <int>`

### ORB
- `--orb-precision <str>` (default: `float32`)
- `--orb-compile`

### MACE
- `--mace-default-dtype <float32|float64>` (default: `float32`)

### AIMNet2
- no extra backend-specific flags in current CLI

## Builtin Test Models

For integration checks without external ML packages:
- `--model builtin:zero`
- `--model builtin:harmonic`
- `--model builtin:harmonic:<k>`

Example:
```text
ml_keywords="--model builtin:zero"
```
