# amber-mlips Options

For most users, defaults are sufficient.

> **Note:** UMA and MACE currently have a dependency conflict (`e3nn`). Use separate environments.

## Wrapper Options

These are consumed by the `amber-mlips` command; all other flags are forwarded to `sander`.

| Option | Description |
|--------|-------------|
| `--sander-bin <path>` | Explicit path to `sander` or `sander.MPI` (default: auto-detect via `AMBERHOME` / `PATH`) |
| `--mpi-bin <path>` | MPI launcher for `--mm-ranks > 1` (default: auto-detect `mpirun` / `mpiexec`) |
| `--mm-ranks <int>` | MPI rank count for MM side (default: `1`) |
| `--keep-transformed-input` | Save transformed mdin as `<input>.amber_mlips.qc.in` for inspection |
| `--dry-run` | Print resolved command without running `sander` |
| `--debug` | Verbose wrapper and shim logs |
| `-h`, `--help` | Print help |

Typical use:
```bash
amber-mlips [wrapper-options] -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 ...
```

## Input File (`&qmmm`)

Plugin-specific fields in `&qmmm`:

| Field | Description |
|-------|-------------|
| `qm_theory` | Backend selector: `uma`, `orb`, `mace`, or `aimnet2` (required) |
| `ml_keywords` | Backend option string (see below) |

All other `&qmmm` fields (`qmmask`, `qmcharge`, `spin`, `qmcut`, etc.) follow native AMBER behavior.

On launch, `amber-mlips` transforms the input:
- `qm_theory` → `'EXTERN'`
- `ml_keywords` is removed (consumed by shim)
- `qm_ewald=0` and `qmgb=0` are forced (EXTERN constraint)
- A generated `&qc` block is appended

## `ml_keywords` — Common Options

`ml_keywords` is parsed with shell token rules and passed to the internal `qchem` shim.

| Option | Description |
|--------|-------------|
| `--model <name>` | Model name, alias, or local path |
| `--device auto\|cpu\|cuda` | Compute device (default: `auto`) |
| `--embedcharge` | Enable xTB point-charge embedding correction |
| `--solvent <name>` | xTB implicit solvent name (default: `none`). E.g., `water`, `methanol`, `dmso` |
| `--solvent-model <alpb\|cpcmx>` | Implicit solvent model (default: `alpb`) |
| `--xtb-cmd <path>` | xTB executable (default: `xtb`) |
| `--xtb-acc <float>` | xTB accuracy parameter (default: `0.2`) |
| `--xtb-ncores <int>` | CPU cores for xTB (default: `4`) |
| `--xtb-workdir <tmp\|path>` | xTB scratch directory (default: `tmp`) |
| `--xtb-keep-files` | Keep xTB temporary files for debugging |
| `--debug` | Verbose shim logs |

When `--embedcharge` or `--solvent` is enabled, xTB must be available in the current environment/path.

### ML-Only MD

Set `qmmask='@*'` to compute all atoms with MLIP (no MM). Non-periodic only (`ntb=0`).

| Setting | Value | Reason |
|---------|-------|--------|
| `qmmask` | `'@*'` | ALL atoms → pure ML MD |
| `ntb` | `0` | Non-periodic (PBC ML-only not supported) |
| `cut` | `999.0` | Large cutoff for non-periodic |
| `ntc`, `ntf` | `1`, `1` | No SHAKE constraints |
| `qmshake` | `0` | No SHAKE on QM atoms |
| `qmcut` | `0.0` | No QM/MM cutoff |

## UMA Options

Available models (default: **`uma-s-1p1`**):

| Model | Description |
|-------|-------------|
| `uma-s-1p1` | Small model, fastest while still SOTA on most benchmarks (6.6M/150M active/total params) |
| `uma-s-1p2` | Small model v1.2, ~50% faster & ~40% more accurate on OMol (6.6M/290M active/total params) |
| `uma-m-1p1` | Best across all metrics, slower and more memory-intensive (50M/1.4B active/total params) |

Additional `esen-*` variants are also available. Models are hosted on Hugging Face Hub (`huggingface-cli login` required).

| Option | Description |
|--------|-------------|
| `--uma-task <omol\|omat\|odac\|oc20\|oc25\|omc>` | Task head selector (default: `omol`) |
| `--uma-workers <int>` | Predictor worker count (default: `1`) |

Example:
```text
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1',
```

## ORB Options

Only conservative ORB models are supported. Underscores and dashes are interchangeable (e.g., `orb_v3_conservative_omol` = `orb-v3-conservative-omol`).

Available models (default: **`orb-v3-conservative-omol`**):

| Model | Dataset |
|-------|---------|
| `orb-v3-conservative-omol` | OMol25 (molecules) |
| `orb-v3-conservative-20-omat` | OMAT (materials, max 20 neighbors) |
| `orb-v3-conservative-inf-omat` | OMAT (materials, unlimited neighbors) |
| `orb-v3-conservative-20-mpa` | MPA (materials, max 20 neighbors) |
| `orb-v3-conservative-inf-mpa` | MPA (materials, unlimited neighbors) |

Models are downloaded automatically on first use.

| Option | Description |
|--------|-------------|
| `--orb-precision <str>` | Float precision (default: `float32`) |
| `--orb-compile` | Enable `torch.compile` for the model |

Example:
```text
  qm_theory='orb',
  ml_keywords='--model orb-v3-conservative-omol',
```

## MACE Options

Available models (default: **`MACE-OMOL-0`**):

| Model | Description |
|-------|-------------|
| `MACE-OMOL-0` | OMOL large model for molecules and transition metals |
| `mp:small`, `mp:medium`, `mp:large` | MACE-MP-0 (Materials Project, 89 elements) |
| `mp:medium-0b3` | MACE-MP-0b3, improved high-pressure stability |
| `mp:medium-mpa-0` | MACE-MPA-0, MPTrj + sAlex |
| `mp:small-omat-0`, `mp:medium-omat-0` | MACE-OMAT-0 |
| `mp:mace-matpes-pbe-0` | MACE-MATPES PBE functional |
| `mp:mace-matpes-r2scan-0` | MACE-MATPES r2SCAN functional |
| `mp:mh-0`, `mp:mh-1` | MACE-MH cross-domain (surfaces/bulk/molecules) |
| `off:small`, `off:medium`, `off:large` | MACE-OFF23 for organic molecules |
| `anicc` | ANI-CC model |

The `mp:` prefix selects Materials Project models, `off:` selects organic force field models. A local file path or URL can also be specified. Models are downloaded automatically on first use.

| Option | Description |
|--------|-------------|
| `--mace-default-dtype <float32\|float64>` | Float precision (default: `float32`) |

Example:
```text
  qm_theory='mace',
  ml_keywords='--model MACE-OMOL-0',
```

## AIMNet2 Options

Available models (default: **`aimnet2`**):

| Model | Description |
|-------|-------------|
| `aimnet2` | AIMNet2 base model |
| `aimnet2_b973c` | AIMNet2 with B97-3c functional |
| `aimnet2_2025` | AIMNet2 B97-3c + improved intermolecular interactions |
| `aimnet2nse` | AIMNet2 open-shell model |
| `aimnet2pd` | AIMNet2 for Pd-containing systems |
| `<local_model_path>` | Local checkpoint file path |
| `<https://...>` | Model URL |

Models are downloaded automatically on first use.

No additional backend-specific options.

Example:
```text
  qm_theory='aimnet2',
  ml_keywords='--model aimnet2',
```
