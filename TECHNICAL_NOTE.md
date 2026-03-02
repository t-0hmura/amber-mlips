# Technical Note

## Wrapper Workflow

`amber-mlips` executes the following sequence:

1. Parse wrapper options and collect all forwarded `sander` flags.
2. Locate `-i <mdin>` in forwarded arguments.
3. Transform user mdin (`plugins/mdin_transform.py`).
4. Start internal `amber-mlips-server` (`plugins/genmpi_server.py`).
5. Launch AMBER (`sander`/`sander.MPI`) with transformed mdin.
6. Wait for completion and clean up server + temporary files.

When `--launcher-mode dvm` (or `auto` with PRRTE available), both server and AMBER are launched via `prun` in the same DVM.

## mdin Transform Rules

`transform_mdin_text(...)` enforces a safe EXTERN path:

- Read backend from user `qm_theory` (`uma|orb|mace|aimnet2`).
- Extract `ml_keywords` (kept only for server launch).
- Rewrite `qm_theory -> 'EXTERN'`.
- Force `qm_ewald=0` and `qmgb=0`.
- Remove pre-existing `&genmpi` blocks.
- Append generated `&genmpi`:
  - `method='UMA'|'ORB'|'MACE'|'AIMNET2'`
  - `basis='MLIP'`
  - `ntpr=1`, `debug=0`, `dipole=0`

Only the transformed mdin is passed to AMBER.

## genmpi Server Protocol

`amber-mlips-server` publishes MPI service name `qc_program_port` and serves
AMBER EXTERN requests over an intercommunicator.

Per step it receives:
- QM charge, spin multiplicity
- QM atom count and 2-char atom labels
- QM coordinates
- MM point-charge count, charges, and coordinates

It returns:
- total QM energy (`Eh`)
- QM charges array (currently zeros)
- dipole array (currently zeros)
- QM gradients (`Eh/Bohr`)
- MM point-charge gradients (`Eh/Bohr`)

AMBER QM/MM MD path here is energy/gradient only (no Hessian exchange).

## Unit Conversion

Backends output MLIP units:
- energy: `eV`
- forces: `eV/Ang`

Server returns AMBER EXTERN units after conversion:
- energy: `Eh`
- gradients: `Eh/Bohr`

Implemented in `plugins/mlip_backends.py`:
- `ev_to_ha(...)`
- `forces_ev_ang_to_gradient_ha_bohr(...)`

## Point-Charge Embedding Correction (`--embedcharge`)

When enabled, server adds xTB-based embedding correction:

- `dE = E_xTB(embed) - E_xTB(no-embed)`
- `dF = F_xTB(embed) - F_xTB(no-embed)`

Injected quantities:
- `E_total = E_MLIP + dE`
- `F_total = F_MLIP + dF` (on QM and MM point-charge blocks)

Implementation:
- correction driver: `plugins/xtb_embedcharge_correction.py`
- integration into request loop: `plugins/genmpi_server.py`

Requires xTB executable (`xtb` or `--xtb-cmd`).

## Launcher and MPI Notes

- `direct` mode starts AMBER and server as normal processes.
- `dvm` mode creates PRRTE DVM (`prte`) and launches both via `prun --dvm-uri ...`.
- OpenMPI5 singleton mode may fail around name publish/lookup; DVM mode avoids most of these failures.

Server teardown intentionally avoids strict `MPI_Unpublish_name`/`MPI_Close_port`
cleanup in some DVM environments to prevent noisy OPAL teardown errors after successful runs.
