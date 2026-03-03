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

    server = MLIPServer(
        evaluator=evaluator,
        socket_path=args.server_socket,
        idle_timeout=args.server_idle_timeout,
        parent_pid=args.server_parent_pid,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
