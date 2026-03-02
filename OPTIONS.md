# amber-mlips Options

For most users, defaults are sufficient.

## Wrapper Options (`amber-mlips`)

These options belong to the top-level wrapper command.
All unrecognized flags are forwarded to `sander`/`sander.MPI`.

- `--sander-bin <path>` — Explicit AMBER executable path.
- `--launcher-mode auto|dvm|direct` — Launch strategy for AMBER + server.
- `--mm-ranks <int>` — MPI ranks for `sander` (MM side). Server always uses 1 rank.
- `--keep-transformed-input` — Save transformed mdin as `<input>.amber_mlips.genmpi.in`.
- `--dry-run` — Print server/AMBER commands without executing.
- `--debug` — Verbose wrapper/server logs.
- `-h`, `--help` — Print wrapper help.

Typical use:
```bash
amber-mlips [wrapper-options] -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 ...
```

## User `&qmmm` Fields

`amber-mlips` adds two user-facing fields in `&qmmm`:

- `qm_theory="uma"|"orb"|"mace"|"aimnet2"` (required)
- `ml_keywords="..."` (optional but usually required)

On transform:
- `qm_theory` is rewritten to `'EXTERN'`
- `ml_keywords` is removed before AMBER run
- `qm_ewald` and `qmgb` are forced to `0`
- generated `&genmpi` is appended

## `ml_keywords` (Server Options)

`ml_keywords` is parsed with shell token rules (`shlex.split`) and passed to
internal `amber-mlips-server`.

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

For integration tests without backend packages:
- `--model builtin:zero`
- `--model builtin:harmonic`
- `--model builtin:harmonic:<k>`

Example:
```text
ml_keywords="--model builtin:zero"
```

## Reserved Server Flags

The wrapper controls these internally; do not pass them in `ml_keywords`:
- `--backend`
- `--service-name`
- `--ready-file`

## Launcher Behavior

- `auto`: uses DVM if both `prte` and `prun` are found; otherwise direct mode.
- `dvm`: requires `prte` + `prun`; fails fast if missing.
- `direct`: no DVM, starts server and AMBER directly.
- `--mm-ranks > 1`: requires DVM mode (`auto` with PRRTE available, or `dvm`) and an MPI-capable sander binary.

In OpenMPI5 environments, DVM mode is generally the most robust for name-service operations.
