# ML-Only MD (Full-System MLIP)

Set `qmmask='@*'` to make **all atoms QM**, yielding pure ML molecular dynamics with no MM force field.
AMBER still needs a parm7 topology file, but all MM forces are replaced by MLIP forces.

## PBC (Periodic Boundary, NVT)

For solvated systems with periodic boundary conditions, cell dimensions are automatically read from the `-c` restart file and passed to the MLIP backend. Use a PBC-aware model (e.g., UMA `omat` task):

```text
 &cntrl
  imin=0, irest=1, ntx=5,
  dt=0.001, nstlim=1000,
  ntc=1, ntf=1,
  cut=3.0, ntb=1, ntp=0,
  ntt=3, gamma_ln=5.0, temp0=300.0,
  iwrap=1, ifqnt=1,
 /
 &qmmm
  qmmask='@*',        ! ALL atoms → pure ML MD
  qmcharge=0, spin=1,
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1 --uma-task omat',
  qmshake=0, qmcut=3.0,
 /
```

Key settings: `ntb=1` (NVT), `ntc=1,ntf=1` (no SHAKE), `qmmask='@*'` (all atoms as QM → pure ML MD), `cut=3.0, qmcut=3.0` (must match; sander checks QM region + qmcut < box), `--uma-task omat` (PBC-aware model).

## PBC (Periodic Boundary, NPT)

For constant-pressure simulations, cell dimensions are updated every step by reading the AMBER restart file.
Two additional settings are **required**:

- `ntwr=1` — write restart every step so the MLIP backend sees the updated cell.
- `ntxo=1` — ASCII restart format (the C shim reads the box line from the restart file).

```text
 &cntrl
  imin=0, irest=1, ntx=5,
  dt=0.001, nstlim=1000,
  ntc=1, ntf=1,
  cut=3.0, ntb=2, ntp=1,
  pres0=1.0, taup=2.0,
  ntt=3, gamma_ln=5.0, temp0=300.0,
  ntwr=1, ntxo=1,
  iwrap=1, ifqnt=1,
 /
 &qmmm
  qmmask='@*',
  qmcharge=0, spin=1,
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1 --uma-task omat',
  qmshake=0, qmcut=3.0,
 /
```

See [examples/uma_mlonly_npt.in](examples/uma_mlonly_npt.in).

> **Note:** The cell is read from the restart file written at the end of the *previous* step, so there is a one-step lag in cell updates. This is negligible for typical NPT simulations where box dimensions change slowly.

## Implicit Solvent (Non-Periodic)

For non-periodic systems with xTB implicit solvation (ALPB/CPCMX):

```text
  ml_keywords='--model uma-s-1p1 --solvent water --solvent-model alpb',
```

This computes `dE = E_xTB(solv) - E_xTB(vac)` and adds the correction to MLIP energy and forces. Use `ntb=0` (no PBC). See [examples/uma_mlonly_implicit.in](examples/uma_mlonly_implicit.in).

Available solvents: water, methanol, ethanol, acetonitrile, dmso, dmf, thf, benzene, toluene, hexane, etc. (all xTB ALPB-supported solvents).
