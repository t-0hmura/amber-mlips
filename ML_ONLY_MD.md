# ML-Only MD (Full-System MLIP)

Set `qmmask='@*'` to make **all atoms QM**, yielding pure ML molecular dynamics with no MM force field.
AMBER still needs a parm7 topology file, but all MM forces are replaced by MLIP forces.

> **Note:** ML-only MD is supported only for **non-periodic** systems (`ntb=0`).
> PBC (periodic boundary) ML-only is not supported due to sander's `QM_CHECK_PERIODIC` limitation,
> which assumes the QM region is a subset of the system.

Two modes are available:

| Mode | Solvent | `ml_keywords` |
|------|---------|---------------|
| Gas phase | None | `--model uma-s-1p1` |
| Implicit solvent | xTB ALPB/CPCMX | `--model uma-s-1p1 --solvent water --solvent-model alpb` |

## Gas Phase

For isolated (vacuum) systems:

```text
 &cntrl
  imin=0, irest=0, ntx=1,
  dt=0.001, nstlim=1000,
  ntc=1, ntf=1,
  cut=999.0, ntb=0, ntp=0,
  ntt=3, gamma_ln=5.0, temp0=300.0,
  iwrap=0, ifqnt=1,
 /
 &qmmm
  qmmask='@*',
  qmcharge=0, spin=1,
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1',
  qmshake=0, qmcut=0.0,
 /
```

## Implicit Solvent

For non-periodic systems with xTB implicit solvation (ALPB/CPCMX):

```text
 &cntrl
  imin=0, irest=0, ntx=1,
  dt=0.001, nstlim=1000,
  ntc=1, ntf=1,
  cut=999.0, ntb=0, ntp=0,
  ntt=3, gamma_ln=5.0, temp0=300.0,
  iwrap=0, ifqnt=1,
 /
 &qmmm
  qmmask='@*',
  qmcharge=0, spin=1,
  qm_theory='uma',
  ml_keywords='--model uma-s-1p1 --solvent water --solvent-model alpb',
  qmshake=0, qmcut=0.0,
 /
```

This computes `dE = E_xTB(solv) - E_xTB(vac)` and adds the correction to MLIP energy and forces. See [examples/uma_mlonly_implicit.in](examples/uma_mlonly_implicit.in).

Available solvents: water, methanol, ethanol, acetonitrile, dmso, dmf, thf, benzene, toluene, hexane, etc. (all xTB ALPB-supported solvents).

## Key Settings

- `ntb=0` — non-periodic (required)
- `cut=999.0` — large cutoff for non-periodic systems
- `qmmask='@*'` — all atoms treated as QM
- `qmcut=0.0` — no QM/MM cutoff (all atoms are QM)
