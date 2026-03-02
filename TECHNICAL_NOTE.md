# Technical Note

## Wrapper Workflow

`amber-mlips` executes this sequence:

1. Parse wrapper options and forwarded `sander` flags.
2. Locate `-i <mdin>` in forwarded arguments.
3. Transform user mdin (`plugins/mdin_transform.py`).
4. Stage a private `qchem` shim in a temporary directory.
5. Set runtime env:
   - prepend shim directory to `PATH`
   - export backend/options via `AMBER_MLIPS_BACKEND` and `AMBER_MLIPS_ML_KEYWORDS`
6. Launch AMBER (`sander` or `sander.MPI`) with transformed mdin.
7. Clean temporary transformed input and shim directory.

There is no MPI name-service process and no `genmpi` server in this runtime path.

## mdin Transform Rules

`transform_mdin_text(...)` enforces EXTERN-compatible input:

- Read backend from user `qm_theory` (`uma|orb|mace|aimnet2`).
- Extract `ml_keywords` (used only by shim, not passed to AMBER `&qmmm`).
- Rewrite `qm_theory -> 'EXTERN'`.
- Force `qm_ewald=0` and `qmgb=0`.
- Remove pre-existing `&genmpi` and `&qc` blocks.
- Append generated `&qc` block.

Only transformed mdin is passed to AMBER.

## qchem Shim Interface

`plugins/nonmpi_qc_shim.py` implements AMBER `&qc` executable contract:

- Called as: `qchem <inpfile> <logfile> <savfile>`
- Parses `<inpfile>` sections:
  - `$molecule`
  - optional `$external_charges`
- Evaluates MLIP backend and optional embedcharge correction.
- Writes:
  - `<logfile>` containing `SCF   energy` line (Hartree)
  - `efield.dat` with MM electric fields then QM gradients

AMBER parser side is `qm2_extern_qc_module.F90`.

## Units and Mapping

MLIP backend native outputs:
- energy: `eV`
- forces: `eV/Ang`

AMBER `&qc` reader expects:
- energy in `Eh`
- QM gradient in `Eh/Bohr`
- MM electric field in atomic units

Conversions:
- `E_h = E_eV / 27.211386245988`
- gradient is `-force`
- MM e-field relation used by AMBER: `grad_mm = -E * q_mm`
  - shim writes `E = -grad_mm / q_mm` for `q_mm != 0`

## Embedcharge (`--embedcharge`)

When enabled, shim applies xTB correction:
- `dE = E_xTB(embed) - E_xTB(no-embed)`
- `dF = F_xTB(embed) - F_xTB(no-embed)`

Injected totals:
- `E_total = E_MLIP + dE`
- `F_total = F_MLIP + dF` (QM and MM point-charge blocks)

Requires `xtb` (or explicit `--xtb-cmd`).

## MM MPI Launch

ML path stays non-MPI. MM side launch is independent:
- `--mm-ranks 1` + `--launcher-mode auto` => direct `sander`
- `--mm-ranks > 1` => MPI launcher (`mpirun`/`mpiexec`) + `sander.MPI`
