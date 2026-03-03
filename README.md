# amber-mlips

MLIP (Machine Learning Interatomic Potential) wrapper for **AMBER QM/MM** via `sander` `EXTERN` interface.

Four model families are currently supported:
- **UMA** ([fairchem](https://github.com/facebookresearch/fairchem)) — default model: `uma-s-1p1`
- **ORB** ([orb-models](https://github.com/orbital-materials/orb-models)) — default model: `orb_v3_conservative_omol`
- **MACE** ([mace](https://github.com/ACEsuit/mace)) — default model: `MACE-OMOL-0`
- **AIMNet2** ([aimnetcentral](https://github.com/isayevlab/aimnetcentral)) — default model: `aimnet2`

All backends provide energy and gradient for AMBER QM/MM molecular dynamics and optimization.
An optional **point-charge embedding** correction (`xTB`) is available via `--embedcharge`.

> `amber-mlips` wraps `sander` so that a single command handles everything — no external server or separate process needed.

Requires **Python 3.9** or later and **AMBER** (`sander`).

## Quick Start (Default = UMA)

1. Install PyTorch suitable for your CUDA environment.
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with the UMA backend. For ORB/MACE/AIMNet2, replace `uma` accordingly.
```bash
pip install "amber-mlips[uma]"
```

3. Log in to Hugging Face for UMA model access. (Not required for ORB/MACE/AIMNet2)
```bash
huggingface-cli login
```

4. Prepare an AMBER input file. Only `qm_theory` and `ml_keywords` are plugin-specific; everything else is native AMBER `&qmmm`.
```text
 &cntrl
  imin=0, irest=1, ntx=5,
  nstlim=1000, dt=0.001,
  ntb=0, ntt=3, gamma_ln=5.0,
  ntpr=10, ntwx=10, ntwr=100,
  ifqnt=1,
 /
 &qmmm
  qmmask=':2',
  qmcharge=0,
  spin=1,
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1',
  qmcut=12.0,
  qmshake=0,
  qm_ewald=0,
  qm_pme=0,
 /
```

Other backends:
```text
  qm_theory='orb',    ml_keywords='--model orb-v3',
  qm_theory='mace',   ml_keywords='--model mace-mp-0-large',
  qm_theory='aimnet2', ml_keywords='--model aimnet2',
```

5. Run with standard `sander` flags.
```bash
amber-mlips -O \
  -i mlmm.in -o mlmm.out \
  -p leap.parm7 -c md.rst7 \
  -r mlmm.rst7 -x mlmm.nc -inf mlmm.info
```

## Point-Charge Embedding Correction (xTB)

`--embedcharge` adds an xTB-based correction for electrostatic embedding of MM point charges into the QM region.

Install xTB:
```bash
conda install xtb
```

Use `--embedcharge` in `ml_keywords`:
```text
  ml_keywords='--model uma-s-1p1 --embedcharge',
```

This computes `dE = E_xTB(embed) - E_xTB(no-embed)` and adds the correction to MLIP energy and forces.

## MM MPI Parallelism

The ML evaluation path is always single-process. The MM side (`sander`) can use MPI:

```bash
amber-mlips --mm-ranks 16 -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 -r mlmm.rst7
```

- `--mm-ranks 1` (default): runs `sander` directly.
- `--mm-ranks > 1`: uses `mpirun`/`mpiexec` + `sander.MPI`.

### AMBER EXTERN MPI Bug (Auto-Patched)

AMBER 24 (and earlier) has a bug in `qm2_extern_module.F90` where non-rank-0
MPI tasks return without initialising output arrays, corrupting forces in
multi-rank EXTERN QM/MM runs. This is an **upstream AMBER bug** affecting all
EXTERN backends, not specific to amber-mlips.

When `--mm-ranks > 1` is used, amber-mlips **automatically** handles this:

1. Checks the AMBER source for the bug.
2. If found, builds a patched `sander.MPI` and caches it in `~/.amber-mlips/`.
3. Uses the patched binary for subsequent runs (no rebuild needed).

If a future AMBER version fixes the bug, the patch is skipped automatically.

**Requirements for auto-patching:**
- AMBER source directory accessible (set `AMBERHOME`; source is located via `${AMBERHOME}_src` or sibling directories)
- Fortran compiler used to build AMBER available in `PATH`

If auto-patching cannot locate the source, an error message explains alternatives
(`--mm-ranks 1` or `--sander-bin /path/to/patched/sander.MPI`).

## Installing Model Families

```bash
pip install "amber-mlips[uma]"         # UMA (default)
pip install "amber-mlips[orb]"         # ORB
pip install "amber-mlips[mace]"        # MACE
pip install "amber-mlips[aimnet2]"     # AIMNet2
pip install amber-mlips                # core only (no ML backend)
```

> **Note:** UMA and MACE have a dependency conflict (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/amber-mlips.git
cd amber-mlips
pip install -e ".[uma]"
```

Model download notes:
- **UMA**: Hosted on Hugging Face Hub. Run `huggingface-cli login` once.
- **ORB / MACE / AIMNet2**: Downloaded automatically on first use.

## Examples

Ready-to-run examples are in the [`examples/`](examples/) directory with an alanine dipeptide test system.

```bash
cd examples
./run.sh uma          # UMA
./run.sh orb          # ORB
./run.sh mace         # MACE
./run.sh embedcharge  # UMA + embedcharge
./run.sh minimize     # energy minimization
./run.sh all          # all of the above
```

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- ORB / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace
- AIMNet2: https://github.com/isayevlab/aimnetcentral

## Advanced Options

See [`OPTIONS.md`](OPTIONS.md) for all wrapper and backend-specific options.

## Troubleshooting

- **`amber-mlips` command not found** — Activate the conda/venv environment where the package is installed.
- **`sander` not found** — Set `AMBERHOME` or use `--sander-bin /path/to/sander`.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **MPI errors with `--mm-ranks > 1`** — Ensure `sander.MPI` is built and `mpirun`/`mpiexec` is available. Use `--mpi-bin` to specify explicitly.
- **Works interactively but fails in batch jobs** — Use `--sander-bin` with an absolute path.

## More Docs

- All options: [`OPTIONS.md`](OPTIONS.md)
- Internal architecture: [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md)

## Citation

If you use this package, please cite:

```bibtex
@software{ohmura2026ambermlips,
  author       = {Ohmura, Takuto},
  title        = {amber-mlips},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/t-0hmura/amber-mlips},
  license      = {MIT}
}
```
