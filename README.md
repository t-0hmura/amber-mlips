# amber-mlips

`amber-mlips` is a single-command wrapper for AMBER QM/MM using MLIP backends through `qm_theory=EXTERN` + `&genmpi`.

## Usage

Run with sander-style flags:

```bash
amber-mlips -O \
  -i mlmm.in \
  -o mlmm.out \
  -p leap.parm7 \
  -c md.rst7 \
  -r mlmm.rst7 \
  -ref leap.parm7 \
  -x mlmm.crd \
  -inf mlmm.info
```

## User mdin (`&qmmm`)

In your input file, set:

```text
&qmmm
  qmmask="@1,2,3",
  qmcharge=0,
  spin=1,
  qm_theory="uma",
  ml_keywords="--model uma-s-1p1 --embedcharge",
  mlcut=12.0,
/
```

Supported MLIP `qm_theory` values:
- `uma`
- `orb`
- `mace`
- `aimnet2`

## Internal behavior

`amber-mlips` automatically:
1. rewrites `&qmmm` to use `qm_theory='EXTERN'`,
2. maps `mlcut -> qmcut`,
3. injects `&genmpi`,
4. starts an internal `amber-mlips-server`,
5. launches AMBER/server through a PRRTE DVM when available (`prte` + `prun`),
6. runs `sander.MPI` (or configured sander binary),
7. stops the server.

Note: AMBER EXTERN requires `qm_ewald=0` and `qmgb=0`, so wrapper enforces these values in transformed input.

## Launcher mode

`amber-mlips` supports:
- `--launcher-mode auto` (default): use DVM (`prte`/`prun`) if available, else direct launch.
- `--launcher-mode dvm`: require DVM launch.
- `--launcher-mode direct`: no DVM.

`OpenMPI 5` singleton mode may fail for `MPI_Publish_name`/`MPI_Lookup_name`.
In that case, use `auto` or `dvm` so wrapper runs both server and `sander.MPI` in the same DVM.

## `--embedcharge` behavior

- `ml_keywords` can include `--embedcharge`.
- When enabled, server applies xTB-based point-charge embedding correction
  (`delta_embedcharge_minus_noembed`) and returns corrected QM/MM gradients.
- Requirements:
  - `xtb` executable available in PATH, or pass `--xtb-cmd /path/to/xtb`.
  - Extra xTB options can be passed in `ml_keywords`:
    - `--xtb-cmd`
    - `--xtb-acc`
    - `--xtb-workdir`
    - `--xtb-keep-files`
    - `--xtb-ncores`

Example:
```text
ml_keywords="--model uma-s-1p1 --embedcharge --xtb-cmd xtb --xtb-ncores 4"
```

## Smoke-test model (dependency-free)

For protocol/integration testing without torch models, use:
- `--model builtin:zero`
- `--model builtin:harmonic`
- `--model builtin:harmonic:0.05`

These are for testing and debugging only.

## Install

```bash
pip install .
```

Development / editable install:

```bash
pip install -e . --no-deps
```

For backend extras:

```bash
pip install .[uma]
pip install .[orb]
pip install .[mace]
pip install .[aimnet2]
```

## Troubleshooting

- If AMBER binaries cannot find `libcudart.so.12`, ensure CUDA runtime is on `LD_LIBRARY_PATH`.
- In some PRRTE environments, a non-fatal `OPAL ERROR: Server not available` may appear during teardown after successful completion.
