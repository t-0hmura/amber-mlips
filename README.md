# amber-mlips

MLIP (Machine Learning Interatomic Potential) wrapper for **AMBER QM/MM** via `sander` `EXTERN` interface.

Four model families are currently supported:
- **UMA** ([fairchem](https://github.com/facebookresearch/fairchem)) — default model: `uma-s-1p1`
- **ORB** ([orb-models](https://github.com/orbital-materials/orb-models)) — default model: `orb_v3_conservative_omol`
- **MACE** ([mace](https://github.com/ACEsuit/mace)) — default model: `MACE-OMOL-0`
- **AIMNet2** ([aimnetcentral](https://github.com/isayevlab/aimnetcentral)) — default model: `aimnet2`

All backends provide energy and gradient for AMBER QM/MM molecular dynamics and optimization.
An optional **point-charge embedding** correction with xTB is available via `--embedcharge`.

Requires **Python 3.9** or later and **AmberTools** (`sander`).
AmberTools is free of charge ([GNU GPL](https://ambermd.org/AmberTools.php)); `sander` / `sander.MPI` are [LGPL 2.1](https://biobb-amber.readthedocs.io/en/latest/sander.html).

> If you use Gaussian 16, see also: https://github.com/t-0hmura/g16-mlips
> If you use ORCA, see also: https://github.com/t-0hmura/orca-mlips

## Quick Start (Default = UMA)

0. (Optional) Install AmberTools with MPI support if not already installed.
```bash
conda install conda-forge::ambertools=*=mpi_mpich_*
```
The MPI variant includes both `sander` and `sander.MPI`. If you only need serial execution, `conda install conda-forge::ambertools` (nompi) also works.
You can also [build from source](https://ambermd.org/GetAmber.php) with `-DMPI=TRUE`.

1. (Optional) Install xTB. Only needed for `--embedcharge`.
```bash
conda install conda-forge::xtb
```
You can also [build from source](https://github.com/grimme-lab/xtb).

2. Install PyTorch suitable for your CUDA environment.
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

3. Install the package with the UMA backend. For ORB/MACE/AIMNet2, replace `uma` accordingly.
```bash
pip install "amber-mlips[uma]"
```

4. Log in to Hugging Face for UMA model access. (Not required for ORB/MACE/AIMNet2)
```bash
huggingface-cli login
```

5. Prepare an AMBER input file. Only `qm_theory` and `ml_keywords` are plugin-specific; everything else is native AMBER `&qmmm`.
```text
 &cntrl
  imin=0, irest=0, ntx=1,
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
 /
```

Other backends:
```text
  qm_theory='orb',    ml_keywords='--model orb-v3-conservative-omol',
  qm_theory='mace',   ml_keywords='--model MACE-OMOL-0',
  qm_theory='aimnet2', ml_keywords='--model aimnet2',
```

6. Run with standard `sander` flags.
```bash
amber-mlips -O \
  -i mlmm.in -o mlmm.out \
  -p leap.parm7 -c md.rst7 \
  -r mlmm.rst7 -x mlmm.nc -inf mlmm.info
```

## Point-Charge Embedding Correction (xTB)

`--embedcharge` adds an xTB-based correction for electrostatic embedding of MM point charges into the QM region.

Install xTB (if not already installed in Quick Start step 1):
```bash
conda install conda-forge::xtb
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
- `--mm-ranks > 1`: uses `mpirun`/`mpiexec` + `sander.MPI` (included in the MPI variant installed in Quick Start step 1).

> **Note:** AMBER 24 (and earlier) has a bug in `qm2_extern_module.F90` that corrupts forces in multi-rank EXTERN runs.
> When `--mm-ranks > 1` is used, amber-mlips automatically detects and patches this bug (requires `AMBERHOME` and Fortran compiler).
> See [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md) for details.

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

Ready-to-run examples are in the [`examples/`](examples/) directory with a protein-ligand system (1IL4, 50,387 atoms, 115 QM atoms).

| File | Backend | Description |
|------|---------|-------------|
| `uma.in` | UMA | `uma-s-1p1` |
| `orb.in` | ORB | `orb-v3-conservative-omol` |
| `mace.in` | MACE | `MACE-OMOL-0` |
| `aimnet2.in` | AIMNet2 | `aimnet2` |
| `uma_embedcharge.in` | UMA | `uma-s-1p1` + xTB embedcharge |

Each backend requires its own environment (see [Installing Model Families](#installing-model-families)). Run the example matching your installed backend directly:
```bash
cd examples
amber-mlips -O -i uma.in -o uma.out -p leap.parm7 -c md.rst7 -r uma.rst7
```

## Performance Reference

Benchmark on a protein-ligand system ([1IL4](./examples/1il4.pdb), 50,387 atoms, 115 [ML-region](./examples/1il4_mlregion.pdb) atoms):

| | UMA | UMA + embedcharge |
|---|---|---|
| Model | `uma-s-1p1` | `uma-s-1p1 --embedcharge` |
| Total atoms | 50,387 | 50,387 |
| ML region atoms | 115 | 115 |
| dt | 0.0005 ps | 0.0005 ps |
| Per step | ~135 ms | ~579 ms |
| Speed | ~321 ps/day | ~75 ps/day |

Environment: AMD Ryzen 7950X3D / 4.20 GHz (32 threads) + RTX 5080 (VRAM 16 GB), RAM 128 GB.
`--mm-ranks 16` used for MM MPI parallelism.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- ORB / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace
- AIMNet2: https://github.com/isayevlab/aimnetcentral

## Advanced Options

See [`OPTIONS.md`](OPTIONS.md) for all wrapper and backend-specific options.
For internal architecture details, see [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).

## Troubleshooting

- **`amber-mlips` command not found** — Activate the conda/venv environment where the package is installed.
- **`sander` not found** — Install AmberTools (`conda install conda-forge::ambertools`), or use `--sander-bin /path/to/sander`.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **MPI errors with `--mm-ranks > 1`** — Ensure `mpirun`/`mpiexec` is available. Use `--mpi-bin` to specify explicitly.
- **Works interactively but fails in batch jobs** — Use `--sander-bin` with an absolute path.

## References

- AMBER24 manual (detailed MD settings): https://ambermd.org/doc12/Amber24.pdf

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
