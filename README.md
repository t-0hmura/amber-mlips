# amber-mlips

MLIP (Machine Learning Interatomic Potential) plugins for **AMBER QM/MM** through
`qm_theory='EXTERN'` + `&genmpi`, exposed as a single `sander`-style command.

Four model families are currently supported:
- **UMA** (fairchem) — default model: `uma-s-1p1`
- **ORB** (orb-models) — default model: `orb-v3-conservative-omol`
- **MACE** (mace) — default model: `small`
- **AIMNet2** (aimnet) — default model: `aimnet2`

The wrapper starts an internal model server automatically, transforms AMBER input internally, runs AMBER, then shuts the server down.

Requires **Python 3.9+** and an AMBER installation with `sander`/`sander.MPI`.

## Quick Start

1. Install PyTorch suitable for your CUDA/runtime.
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install `amber-mlips` with your backend profile.
```bash
pip install "amber-mlips[uma]"
```
For ORB/MACE/AIMNet2, use `amber-mlips[orb]`, `amber-mlips[mace]`, `amber-mlips[aimnet2]`.

3. Prepare your AMBER input (`mlmm.in`) with plugin-style `&qmmm` fields:
```text
&qmmm
  qmmask="@1,2,3",
  qmcharge=0,
  spin=1,
  qm_theory="uma",
  ml_keywords="--model uma-s-1p1",
  qmcut=12.0,
/
```

4. Run with standard `sander` flags:
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

MM CPU-side MPI parallelism:
```bash
amber-mlips --mm-ranks 16 -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 -r mlmm.rst7
```
`--mm-ranks` changes only the `sander.MPI` rank count.
The internal ML server is always launched with 1 rank.

## Input Style (User View)

In `&qmmm`, `amber-mlips` expects:
- `qm_theory="uma"|"orb"|"mace"|"aimnet2"`
- `ml_keywords="..."` (options passed to internal MLIP server)

All other AMBER control fields are kept as AMBER input.

## Internal Transform (AMBER View)

`amber-mlips` rewrites `mlmm.in` before launching AMBER:
- `qm_theory` -> `'EXTERN'`
- remove `ml_keywords`
- force `qm_ewald=0` and `qmgb=0` (AMBER EXTERN requirement)
- remove existing `&genmpi` blocks
- append generated `&genmpi` with backend-specific `method`

You can inspect transformed input with:
```bash
amber-mlips --keep-transformed-input ...
```

## `ml_keywords` and `--embedcharge`

`ml_keywords` is parsed like shell CLI tokens and forwarded to the internal
`amber-mlips-server` process.

Example:
```text
ml_keywords="--model uma-s-1p1 --embedcharge --xtb-cmd xtb --xtb-ncores 4"
```

`--embedcharge` enables xTB point-charge embedding correction using MM point charges from the AMBER EXTERN interface.

For `--embedcharge`, xTB must be available (`xtb` in `PATH` or `--xtb-cmd /path/to/xtb`).

## Launcher Modes

- `--launcher-mode auto` (default): use PRRTE DVM (`prte` + `prun`) when available, else direct launch.
- `--launcher-mode dvm`: require DVM.
- `--launcher-mode direct`: no DVM.

`auto`/`dvm` is recommended for OpenMPI5 environments where singleton naming can fail.

## Smoke-Test Models

For protocol/integration checks without heavy ML dependencies:
- `--model builtin:zero`
- `--model builtin:harmonic`
- `--model builtin:harmonic:0.05`

These are for testing/debugging only.

## Installing Model Families

```bash
pip install "amber-mlips[uma]"      # UMA
pip install "amber-mlips[orb]"      # ORB
pip install "amber-mlips[mace]"     # MACE
pip install "amber-mlips[aimnet2]"  # AIMNet2
pip install amber-mlips              # core only
```

Local editable install:
```bash
cd amber-mlips
pip install -e . --no-deps
```

Model download notes:
- **UMA**: Hugging Face access may require `huggingface-cli login`.
- **ORB / MACE / AIMNet2**: downloaded automatically on first use.

## Advanced Options

See [`OPTIONS.md`](OPTIONS.md) for wrapper and backend/server options.

Implementation details are summarized in [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).

Command entry points:
- `amber-mlips` (single-command AMBER wrapper)
- `amber-mlips-server` (internal genmpi server; normally not run directly)

## Troubleshooting

- **`Internal MLIP server exited early`**:
  - Check backend dependencies (torch/fairchem/orb-models/mace/aimnet).
  - Use `--dry-run --debug` to inspect generated commands.
- **MPI name service errors (`Publish_name` / `Lookup_name`)**:
  - Use `--launcher-mode auto` or `--launcher-mode dvm`.
- **`qm_theory='...' is not supported`**:
  - Use one of `uma`, `orb`, `mace`, `aimnet2` in user `&qmmm`.
- **xTB error with `--embedcharge`**:
  - Confirm `xtb` is installed and reachable; set `--xtb-cmd` explicitly.
- **CUDA/OpenMPI runtime library errors**:
  - Ensure your module/conda environment exports required runtime libraries in `LD_LIBRARY_PATH`.
