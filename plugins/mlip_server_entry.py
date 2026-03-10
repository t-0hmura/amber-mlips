#!/usr/bin/env python3
"""Entry point for the persistent MLIP model server.

Spawned by cli_amber.py as a background process.  Loads the model once
and serves evaluation requests over a Unix domain socket.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from .mlip_server import MLIPServer
from .nonmpi_qc_shim import _create_evaluator, _parse_keywords


def main(argv=None):
    parser = argparse.ArgumentParser(prog="amber-mlips-server")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--ml-keywords", default="")
    parser.add_argument("--server-socket", required=True)
    parser.add_argument("--server-parent-pid", type=int, default=None)
    parser.add_argument("--server-idle-timeout", type=int, default=3600)
    parser.add_argument("--server-cell", default=None,
                        help="Cell parameters: Lx,Ly,Lz,alpha,beta,gamma (for PBC)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    kw_args = _parse_keywords(backend=args.backend, ml_keywords=args.ml_keywords)

    print(
        "[amber-mlips-server] Loading model: backend={}, ml_keywords={}".format(
            args.backend, args.ml_keywords or "(empty)"
        ),
        file=sys.stderr,
        flush=True,
    )

    evaluator = _create_evaluator(kw_args)

    print(
        "[amber-mlips-server] Model loaded: {}".format(evaluator.__class__.__name__),
        file=sys.stderr,
        flush=True,
    )

    # Pass embedcharge options to server if --embedcharge is active.
    embedcharge_opts = None
    if getattr(kw_args, "embedcharge", False):
        embedcharge_opts = {
            "xtb_cmd": getattr(kw_args, "xtb_cmd", "xtb"),
            "xtb_acc": getattr(kw_args, "xtb_acc", 0.2),
            "xtb_workdir": getattr(kw_args, "xtb_workdir", "tmp"),
            "xtb_keep_files": getattr(kw_args, "xtb_keep_files", False),
            "ncores": getattr(kw_args, "xtb_ncores", 4),
        }

    # Pass solvent options to server if --solvent is active.
    solvent_opts = None
    solvent_name = getattr(kw_args, "solvent", "none")
    if solvent_name and str(solvent_name).strip().lower() != "none":
        solvent_opts = {
            "solvent": str(solvent_name),
            "solvent_model": getattr(kw_args, "solvent_model", "alpb"),
            "xtb_cmd": getattr(kw_args, "xtb_cmd", "xtb"),
            "xtb_acc": getattr(kw_args, "xtb_acc", 0.2),
            "xtb_workdir": getattr(kw_args, "xtb_workdir", "tmp"),
            "xtb_keep_files": getattr(kw_args, "xtb_keep_files", False),
            "ncores": getattr(kw_args, "xtb_ncores", 4),
        }

    # Parse cell info for PBC (passed from cli_amber.py at startup).
    cell_ang = None
    if args.server_cell:
        try:
            import numpy as _np
            vals = [float(v) for v in args.server_cell.split(",")]
            if len(vals) == 6:
                cell_ang = _np.array(vals, dtype=_np.float64).reshape(2, 3)
                print(
                    "[amber-mlips-server] PBC cell: lengths=[{:.4f},{:.4f},{:.4f}] "
                    "angles=[{:.2f},{:.2f},{:.2f}]".format(*vals),
                    file=sys.stderr, flush=True,
                )
        except Exception as exc:
            print(
                "[amber-mlips-server] WARNING: Failed to parse --server-cell: {}".format(exc),
                file=sys.stderr, flush=True,
            )

    # Warn if PBC and implicit solvent are both active (physically unusual).
    if cell_ang is not None and solvent_opts is not None:
        print(
            "[amber-mlips-server] WARNING: PBC cell and implicit solvent are both active. "
            "PBC typically implies explicit solvent; implicit solvent is for non-periodic systems.",
            file=sys.stderr, flush=True,
        )

    server = MLIPServer(
        evaluator=evaluator,
        socket_path=args.server_socket,
        idle_timeout=args.server_idle_timeout,
        parent_pid=args.server_parent_pid,
        embedcharge_opts=embedcharge_opts,
        solvent_opts=solvent_opts,
        cell=cell_ang,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
