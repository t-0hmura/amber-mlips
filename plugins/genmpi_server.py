#!/usr/bin/env python3
"""genmpi-compatible MLIP service for AMBER EXTERN QM/MM.

This process publishes `qc_program_port` and serves requests from AMBER's
`&genmpi` EXTERN interface.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import traceback

import numpy as np

from .mlip_backends import (
    AIMNet2Evaluator,
    BackendError,
    MACEEvaluator,
    OrbMolEvaluator,
    UMAEvaluator,
    ev_to_ha,
    forces_ev_ang_to_gradient_ha_bohr,
)
from .xtb_alpb_correction import XTBError
from .xtb_embedcharge_correction import delta_embedcharge_minus_noembed


class GenMPIError(RuntimeError):
    """Raised for protocol or runtime failures in the genmpi server."""


class _BuiltinEvaluator(object):
    """Small dependency-free evaluator used for smoke/integration tests.

    Model spec:
    - builtin:zero
    - builtin:harmonic
    - builtin:harmonic:<k>
    """

    def __init__(self, model_spec):
        spec = str(model_spec or "builtin:zero").strip().lower()
        self.kind = "zero"
        self.k = 0.05

        if spec.startswith("builtin:harmonic"):
            self.kind = "harmonic"
            parts = spec.split(":")
            if len(parts) >= 3:
                try:
                    self.k = float(parts[2])
                except Exception:
                    self.k = 0.05
            if self.k <= 0.0:
                self.k = 0.05
        elif spec.startswith("builtin:zero"):
            self.kind = "zero"
        else:
            # Keep behavior deterministic for unknown builtin aliases.
            self.kind = "zero"

    def evaluate(
        self,
        symbols,
        coords_ang,
        charge,
        multiplicity,
        need_forces,
        need_hessian,
        hessian_mode,
        hessian_step,
    ):
        _ = (symbols, charge, multiplicity, need_hessian, hessian_mode, hessian_step)
        coords = np.asarray(coords_ang, dtype=np.float64).reshape(-1, 3)
        nat = coords.shape[0]
        if self.kind == "harmonic":
            # E = 1/2 k r^2, F = -k r
            energy_ev = 0.5 * self.k * float(np.sum(coords * coords))
            forces = -self.k * coords
        else:
            energy_ev = 0.0
            forces = np.zeros((nat, 3), dtype=np.float64)
        return float(energy_ev), np.asarray(forces, dtype=np.float64), None


def _normalize_symbol(sym_text):
    txt = str(sym_text or "").strip()
    if not txt:
        return "X"
    if len(txt) == 1:
        return txt.upper()
    return txt[0].upper() + txt[1:].lower()


def _decode_atom_types(atom_type_bytes, natoms):
    if int(natoms) <= 0:
        return []
    raw = bytes(atom_type_bytes)
    out = []
    for i in range(int(natoms)):
        block = raw[2 * i : 2 * i + 2]
        try:
            sym = block.decode("ascii", errors="ignore")
        except Exception:
            sym = "X"
        out.append(_normalize_symbol(sym))
    return out


def _create_evaluator(args):
    backend = str(args.backend).lower()
    model = str(args.model or "").strip()

    if model.lower().startswith("builtin:"):
        return _BuiltinEvaluator(model)

    if backend == "uma":
        return UMAEvaluator(
            model=model or "uma-s-1p1",
            task=args.uma_task,
            device=args.device,
            workers=max(1, int(args.uma_workers)),
            workers_per_node=None,
            max_neigh=None,
            radius=None,
            r_edges=False,
            otf_graph=True,
        )

    if backend == "orb":
        return OrbMolEvaluator(
            model=model or "orb-v3-conservative-omol",
            device=args.device,
            precision=args.orb_precision,
            compile_model=bool(args.orb_compile),
            loader_kwargs=None,
            calc_kwargs=None,
        )

    if backend == "mace":
        return MACEEvaluator(
            model=model or "small",
            device=args.device,
            default_dtype=args.mace_default_dtype,
            calc_kwargs=None,
        )

    if backend == "aimnet2":
        return AIMNet2Evaluator(
            model=model or "aimnet2",
            device=args.device,
            calc_kwargs=None,
        )

    raise GenMPIError("Unsupported backend: {}".format(backend))


def _publish_name(MPI, service_name, port_name):
    try:
        MPI.Publish_name(service_name, port_name)
        return
    except TypeError:
        pass
    MPI.Publish_name(service_name, MPI.INFO_NULL, port_name)


def _accept(COMM_SELF, port_name, MPI):
    try:
        return COMM_SELF.Accept(port_name, info=MPI.INFO_NULL, root=0)
    except TypeError:
        return COMM_SELF.Accept(port_name, MPI.INFO_NULL, 0)


def _recv_initial_job_info(intercomm, MPI, debug=False):
    """Receive AMBER's fixed-size initial namelist payload.

    In qm2_extern_genmpi_module.F90, AMBER sends:
      character(len=256) :: dbuffer(128)
      MPI_Send(..., count=256*128, MPI_CHARACTER, ...)
    """
    nchar = 256 * 128
    payload = np.empty(nchar, dtype=np.byte)
    intercomm.Recv([payload, MPI.CHAR], source=0, tag=MPI.ANY_TAG)

    if not debug:
        return

    raw = payload.tobytes().decode("ascii", errors="ignore")
    rows = []
    for i in range(128):
        row = raw[i * 256 : (i + 1) * 256].strip()
        if row:
            rows.append(row)
    preview = rows[:8]
    print(
        "[amber-mlips-server] Received initial job info rows: {}".format(
            "; ".join(preview) if preview else "(empty)"
        ),
        file=sys.stderr,
        flush=True,
    )


def run_server(args):
    try:
        from mpi4py import MPI
    except Exception as exc:
        raise GenMPIError(
            "mpi4py is required for genmpi server. Install with: pip install mpi4py"
        ) from exc

    evaluator = _create_evaluator(args)

    service_name = str(args.service_name).strip() or "qc_program_port"
    ready_file = str(args.ready_file).strip() if args.ready_file else None
    debug = bool(args.debug)

    port_name = None
    intercomm = None

    try:
        port_name = MPI.Open_port()
        _publish_name(MPI, service_name, port_name)

        if ready_file:
            with open(os.path.abspath(ready_file), "w") as handle:
                handle.write("ready\n")

        if debug:
            print(
                "[amber-mlips-server] Published {} on {}".format(service_name, port_name),
                file=sys.stderr,
                flush=True,
            )

        intercomm = _accept(MPI.COMM_SELF, port_name, MPI)

        if debug:
            print("[amber-mlips-server] AMBER connected.", file=sys.stderr, flush=True)

        _recv_initial_job_info(intercomm, MPI, debug=debug)

        status = MPI.Status()

        while True:
            charge = np.empty(1, dtype=np.int32)
            intercomm.Recv([charge, MPI.INT], source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == 0:
                if debug:
                    print("[amber-mlips-server] Received finalize tag=0.", file=sys.stderr, flush=True)
                break

            spinmult = np.empty(1, dtype=np.int32)
            intercomm.Recv([spinmult, MPI.INT], source=0, tag=MPI.ANY_TAG)

            nqmatoms = np.empty(1, dtype=np.int32)
            intercomm.Recv([nqmatoms, MPI.INT], source=0, tag=MPI.ANY_TAG)
            nq = int(nqmatoms[0])

            atom_types = np.empty(2 * nq, dtype=np.byte)
            if nq > 0:
                intercomm.Recv([atom_types, MPI.CHAR], source=0, tag=MPI.ANY_TAG)
            symbols = _decode_atom_types(atom_types.tobytes(), nq)

            qm_flat = np.empty(3 * nq, dtype=np.float64)
            if nq > 0:
                intercomm.Recv([qm_flat, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG)
            qm_coords = qm_flat.reshape(nq, 3)

            nclatoms = np.empty(1, dtype=np.int32)
            intercomm.Recv([nclatoms, MPI.INT], source=0, tag=MPI.ANY_TAG)
            ncl = int(nclatoms[0])

            mm_charges = np.empty(ncl, dtype=np.float64)
            if ncl > 0:
                intercomm.Recv([mm_charges, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG)

            mm_flat = np.empty(3 * ncl, dtype=np.float64)
            if ncl > 0:
                intercomm.Recv([mm_flat, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG)
            mm_coords = mm_flat.reshape(ncl, 3)

            # Evaluate (QM only in release 1).
            energy_ev, forces_ev_ang, _ = evaluator.evaluate(
                symbols=symbols,
                coords_ang=qm_coords,
                charge=int(charge[0]),
                multiplicity=int(spinmult[0]),
                need_forces=True,
                need_hessian=False,
                hessian_mode="analytical",
                hessian_step=1.0e-3,
            )
            forces_q = np.asarray(forces_ev_ang, dtype=np.float64).reshape(nq, 3)
            forces_m = np.zeros((ncl, 3), dtype=np.float64)

            if bool(args.embedcharge) and ncl > 0:
                try:
                    de_embed, df_full, _ = delta_embedcharge_minus_noembed(
                        symbols=symbols,
                        coords_q_ang=qm_coords,
                        mm_coords_ang=mm_coords,
                        mm_charges=mm_charges,
                        charge=int(charge[0]),
                        multiplicity=int(spinmult[0]),
                        need_forces=True,
                        need_hessian=False,
                        xtb_cmd=args.xtb_cmd,
                        xtb_acc=float(args.xtb_acc),
                        xtb_workdir=args.xtb_workdir,
                        xtb_keep_files=bool(args.xtb_keep_files),
                        ncores=int(args.xtb_ncores),
                        hessian_step=1.0e-3,
                    )
                except XTBError as exc:
                    raise GenMPIError(
                        "Embedcharge correction failed. "
                        "Set --xtb-cmd correctly or disable --embedcharge. Details: {}".format(exc)
                    ) from exc

                energy_ev += float(de_embed)
                df_full = np.asarray(df_full, dtype=np.float64).reshape(nq + ncl, 3)
                forces_q = forces_q + df_full[:nq, :]
                if ncl > 0:
                    forces_m = forces_m + df_full[nq:, :]

            energy_ha = np.asarray([ev_to_ha(energy_ev)], dtype=np.float64)
            dxyzqm = np.asarray(
                forces_ev_ang_to_gradient_ha_bohr(forces_q),
                dtype=np.float64,
            )
            dxyzcl = np.asarray(
                forces_ev_ang_to_gradient_ha_bohr(forces_m),
                dtype=np.float64,
            )
            qmcharges = np.zeros(nq, dtype=np.float64)
            dipole = np.zeros(4, dtype=np.float64)

            if debug:
                print(
                    "[amber-mlips-server] step: nq={} ncl={} E={:.10f} Eh".format(
                        nq, ncl, float(energy_ha[0])
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            intercomm.Send([energy_ha, MPI.DOUBLE], dest=0, tag=1)
            intercomm.Send([qmcharges, MPI.DOUBLE], dest=0, tag=1)
            intercomm.Send([dipole, MPI.DOUBLE], dest=0, tag=1)
            if nq > 0:
                intercomm.Send([dxyzqm, MPI.DOUBLE], dest=0, tag=1)
            else:
                intercomm.Send([np.empty(0, dtype=np.float64), MPI.DOUBLE], dest=0, tag=1)
            if ncl > 0:
                intercomm.Send([dxyzcl, MPI.DOUBLE], dest=0, tag=1)
            else:
                intercomm.Send([np.empty(0, dtype=np.float64), MPI.DOUBLE], dest=0, tag=1)

    except BackendError:
        raise
    except Exception as exc:
        raise GenMPIError("genmpi server failed: {}".format(exc)) from exc
    finally:
        if intercomm is not None:
            try:
                intercomm.Disconnect()
            except Exception:
                pass
        # NOTE:
        # In PRRTE DVM-backed runs, both MPI_Unpublish_name and MPI_Close_port
        # can emit noisy OPAL errors at teardown despite successful completion.
        # The process is short-lived, so we intentionally skip explicit port
        # teardown here and rely on process finalization.


def build_parser():
    parser = argparse.ArgumentParser(
        prog="amber-mlips-server",
        description="Run a genmpi MLIP service for AMBER EXTERN QM/MM.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("uma", "orb", "mace", "aimnet2"),
        help="MLIP backend.",
    )
    parser.add_argument("--model", default=None, help="Model alias/path.")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument(
        "--service-name",
        default="qc_program_port",
        help="MPI published service name (AMBER expects qc_program_port).",
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="Write this file when Publish_name is complete.",
    )
    parser.add_argument("--embedcharge", action="store_true", help="Enable xTB point-charge embedding correction.")
    parser.add_argument("--xtb-cmd", default="xtb", help="xTB executable for --embedcharge correction.")
    parser.add_argument("--xtb-acc", type=float, default=0.2, help="xTB --acc value for embedcharge correction.")
    parser.add_argument("--xtb-workdir", default="tmp", help="xTB scratch base dir for embedcharge correction.")
    parser.add_argument("--xtb-keep-files", action="store_true", help="Keep xTB scratch directories for debugging.")
    parser.add_argument("--xtb-ncores", type=int, default=1, help="xTB OMP threads for embedcharge correction.")
    parser.add_argument("--debug", action="store_true")

    # UMA
    parser.add_argument("--uma-task", default="omol")
    parser.add_argument("--uma-workers", type=int, default=1)

    # ORB
    parser.add_argument("--orb-precision", default="float32")
    parser.add_argument("--orb-compile", action="store_true")

    # MACE
    parser.add_argument("--mace-default-dtype", default="float32")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_server(args)
        return 0
    except Exception as exc:
        print("[amber-mlips-server] ERROR: {}".format(exc), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
