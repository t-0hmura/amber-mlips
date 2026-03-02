#!/usr/bin/env python3
"""Single-command amber-mlips wrapper.

Usage pattern mirrors sander flags:

    amber-mlips -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 ...

Internally:
1) transforms `mlmm.in` from backend-style qmmm into EXTERN+genmpi input,
2) starts an internal genmpi MLIP server,
3) runs sander/sander.MPI with transformed input,
4) shuts down server.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from .mdin_transform import InputTransformError, transform_mdin_text


class AmberMLIPSError(RuntimeError):
    """Raised for wrapper-level runtime failures."""


def _build_runtime_env():
    """Return subprocess environment with practical OpenMPI defaults."""
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_mca_base_component_show_load_errors", "none")
    env.setdefault("OMPI_MCA_opal_warn_on_missing_libcuda", "0")
    if "OMPI_MCA_opal_cuda_support" not in env and not os.path.exists("/dev/nvidiactl"):
        env["OMPI_MCA_opal_cuda_support"] = "0"
    return env


def _resolve_sander_bin(user_choice):
    if user_choice:
        return user_choice

    candidates = []
    amberhome = os.environ.get("AMBERHOME", "").strip()
    if amberhome:
        candidates.extend(
            [
                os.path.join(amberhome, "bin", "sander.MPI"),
                os.path.join(amberhome, "bin", "sander"),
            ]
        )

    candidates.extend(
        [
            "/home/apps/amber24/bin/sander.MPI",
            "/home/apps/amber24/bin/sander",
        ]
    )

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    for name in ("sander.MPI", "sander"):
        found = shutil.which(name)
        if found:
            return found

    raise AmberMLIPSError(
        "Could not locate sander executable. Set AMBERHOME or use --sander-bin PATH."
    )


def _choose_launcher(mode):
    mode = str(mode or "auto").strip().lower()
    if mode not in {"auto", "dvm", "direct"}:
        raise AmberMLIPSError("Unknown --launcher-mode '{}'.".format(mode))

    if mode == "direct":
        return "direct", None, None

    prte = shutil.which("prte")
    prun = shutil.which("prun")
    if prte and prun:
        return "dvm", prte, prun

    if mode == "dvm":
        missing = []
        if not prte:
            missing.append("prte")
        if not prun:
            missing.append("prun")
        raise AmberMLIPSError(
            "Launcher mode 'dvm' requested but required command(s) missing: {}.".format(
                ", ".join(missing)
            )
        )

    return "direct", None, None


def _wait_for_nonempty_file(path, timeout_sec):
    t0 = time.time()
    while True:
        if os.path.isfile(path):
            try:
                with open(path, "r") as handle:
                    txt = handle.read().strip()
                if txt:
                    return txt
            except OSError:
                pass
        if (time.time() - t0) > float(timeout_sec):
            raise AmberMLIPSError("Timed out waiting for file '{}'.".format(path))
        time.sleep(0.1)


def _start_dvm(prte_bin, debug=False, env=None):
    uri_fd, uri_file = tempfile.mkstemp(prefix="amber_mlips_dvm_", suffix=".uri")
    os.close(uri_fd)
    os.unlink(uri_file)

    pid_fd, pid_file = tempfile.mkstemp(prefix="amber_mlips_dvm_", suffix=".pid")
    os.close(pid_fd)
    os.unlink(pid_file)

    cmd = [
        prte_bin,
        "--daemonize",
        "--report-uri",
        uri_file,
        "--report-pid",
        pid_file,
    ]
    subprocess.run(cmd, check=True, env=env)

    uri_text = _wait_for_nonempty_file(uri_file, timeout_sec=20.0)
    uri_line = uri_text.splitlines()[0].strip()

    pid_text = _wait_for_nonempty_file(pid_file, timeout_sec=20.0)
    pid_line = pid_text.splitlines()[0].strip()
    try:
        pid = int(pid_line)
    except Exception as exc:
        raise AmberMLIPSError("Failed to parse DVM pid '{}'.".format(pid_line)) from exc

    if debug:
        print(
            "[amber-mlips] started DVM pid={} uri={}".format(pid, uri_line),
            file=sys.stderr,
        )

    return {
        "cmd": cmd,
        "pid": pid,
        "uri": uri_line,
        "uri_file": uri_file,
        "pid_file": pid_file,
    }


def _stop_dvm(dvm_state, env=None):
    if not dvm_state:
        return

    pid = dvm_state.get("pid")
    uri_file = dvm_state.get("uri_file")

    pterm_bin = shutil.which("pterm")
    if pterm_bin and uri_file and os.path.exists(uri_file):
        try:
            subprocess.run(
                [pterm_bin, "--dvm-uri", "file:{}".format(uri_file)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except Exception:
            pass

    if pid is not None:
        t0 = time.time()
        while (time.time() - t0) < 3.0:
            try:
                os.kill(int(pid), 0)
            except OSError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
            t1 = time.time()
            while (time.time() - t1) < 2.0:
                try:
                    os.kill(int(pid), 0)
                except OSError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass

    for key in ("uri_file", "pid_file"):
        path = dvm_state.get(key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _build_prun_prefix(prun_bin, dvm_uri_file, n_ranks):
    n = int(n_ranks)
    if n < 1:
        raise AmberMLIPSError("--mm-ranks must be >= 1.")
    return [
        prun_bin,
        "--dvm-uri",
        "file:{}".format(dvm_uri_file),
        "-n",
        str(n),
    ]


def _extract_input_path(argv):
    for i, token in enumerate(argv):
        if token == "-i":
            if i + 1 >= len(argv):
                raise AmberMLIPSError("-i flag was provided without a file path.")
            return i, None, argv[i + 1]
        if token.startswith("-i") and len(token) > 2:
            return i, token[2:], token[2:]
    raise AmberMLIPSError("amber-mlips requires -i <mdin> so the wrapper can transform &qmmm.")


def _replace_input_path(argv, idx, inline_path, new_path):
    out = list(argv)
    if inline_path is None:
        out[idx + 1] = new_path
    else:
        out[idx] = "-i{}".format(new_path)
    return out


def _server_keyword_args(ml_keywords):
    keyword_args = []
    if ml_keywords:
        keyword_args = shlex.split(str(ml_keywords))

    forbidden = {"--backend", "--service-name", "--ready-file"}
    for tok in keyword_args:
        if tok in forbidden:
            raise AmberMLIPSError(
                "ml_keywords contains reserved option '{}'; this is controlled internally.".format(tok)
            )
    return keyword_args


def _build_server_cmd(backend, ready_file, ml_keywords, debug=False):
    keyword_args = _server_keyword_args(ml_keywords)
    cmd = [
        sys.executable,
        "-m",
        "amber_mlips_plugins.genmpi_server",
        "--backend",
        backend,
        "--service-name",
        "qc_program_port",
        "--ready-file",
        ready_file,
    ]
    if debug:
        cmd.append("--debug")
    cmd.extend(keyword_args)
    return cmd


def _start_server(backend, ml_keywords, debug=False, launcher_prefix=None, env=None):
    ready_fd, ready_file = tempfile.mkstemp(prefix="amber_mlips_ready_", suffix=".flag")
    os.close(ready_fd)
    os.unlink(ready_file)

    base_cmd = _build_server_cmd(
        backend=backend,
        ready_file=ready_file,
        ml_keywords=ml_keywords,
        debug=debug,
    )
    cmd = list(launcher_prefix or []) + base_cmd

    proc = subprocess.Popen(cmd, env=env)

    timeout_sec = 30.0
    t0 = time.time()
    while True:
        if os.path.exists(ready_file):
            return proc, ready_file, cmd

        if proc.poll() is not None:
            raise AmberMLIPSError(
                "Internal MLIP server exited early with code {}.".format(proc.returncode)
            )

        if (time.time() - t0) > timeout_sec:
            proc.terminate()
            raise AmberMLIPSError(
                "Timed out waiting for MLIP server to publish qc_program_port."
            )

        time.sleep(0.1)


def _stop_server(proc):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def _write_transformed_input(input_path, transformed_text, keep_file):
    src_abs = os.path.abspath(input_path)

    if keep_file:
        out_path = src_abs + ".amber_mlips.genmpi.in"
        with open(out_path, "w") as handle:
            handle.write(transformed_text)
        return out_path, None

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="amber_mlips_",
        suffix=".in",
        delete=False,
    )
    try:
        tmp.write(transformed_text)
    finally:
        tmp.close()
    return tmp.name, tmp.name


def build_parser():
    parser = argparse.ArgumentParser(
        prog="amber-mlips",
        description="Run AMBER QM/MM with MLIP backend via internal EXTERN+genmpi bridge.",
        add_help=False,
    )
    parser.add_argument("--sander-bin", default=None, help="Path to sander executable (default: auto-detect sander.MPI).")
    parser.add_argument(
        "--launcher-mode",
        default="auto",
        choices=("auto", "dvm", "direct"),
        help="Execution launcher: auto(detect prte/prun), dvm(always prte+prun), or direct(no DVM).",
    )
    parser.add_argument(
        "--mm-ranks",
        type=int,
        default=1,
        help="MPI ranks for sander (MM side). ML server always runs with 1 rank.",
    )
    parser.add_argument("--keep-transformed-input", action="store_true", help="Keep transformed mdin as '<input>.amber_mlips.genmpi.in'.")
    parser.add_argument("--dry-run", action="store_true", help="Transform input and print commands, but do not run sander.")
    parser.add_argument("--debug", action="store_true", help="Verbose wrapper/server logs.")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message.")
    return parser


def _print_help(parser):
    parser.print_help(sys.stderr)
    print(
        "\nPass all standard sander flags after wrapper options, e.g.\n"
        "  amber-mlips --mm-ranks 16 -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 -r mlmm.rst7\n"
        "\nqmmm requirements in the user mdin:\n"
        "  qm_theory = \"uma\"|\"orb\"|\"mace\"|\"aimnet2\"\n"
        "  ml_keywords = \"--model ... [backend options]\"\n"
        "  (other qmmm fields follow native AMBER behavior)\n",
        file=sys.stderr,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    ns, forward = parser.parse_known_args(argv)

    if ns.help:
        _print_help(parser)
        return 0

    if not forward:
        _print_help(parser)
        return 2

    server_proc = None
    dvm_state = None
    ready_file = None
    generated_input = None
    subprocess_env = _build_runtime_env()

    try:
        i_idx, inline_i, user_input = _extract_input_path(forward)

        with open(user_input, "r") as handle:
            original_text = handle.read()

        transformed = transform_mdin_text(original_text)

        for warn in transformed.warnings:
            print("[amber-mlips] WARNING: {}".format(warn), file=sys.stderr)

        transformed_path, generated_input = _write_transformed_input(
            user_input,
            transformed.transformed_text,
            keep_file=bool(ns.keep_transformed_input),
        )

        forward_exec = _replace_input_path(forward, i_idx, inline_i, transformed_path)
        sander_bin = _resolve_sander_bin(ns.sander_bin)
        mm_ranks = int(ns.mm_ranks)
        if mm_ranks < 1:
            raise AmberMLIPSError("--mm-ranks must be >= 1.")

        launcher_mode, prte_bin, prun_bin = _choose_launcher(ns.launcher_mode)
        if launcher_mode == "direct" and mm_ranks > 1:
            raise AmberMLIPSError(
                "--mm-ranks > 1 requires DVM launch. Use --launcher-mode auto or dvm."
            )
        if mm_ranks > 1:
            sander_name = os.path.basename(str(sander_bin)).lower()
            if ".mpi" not in sander_name:
                raise AmberMLIPSError(
                    "--mm-ranks > 1 requires an MPI-capable sander binary (e.g., sander.MPI). "
                    "Current binary: {}".format(sander_bin)
                )
        if ns.debug:
            print("[amber-mlips] backend: {}".format(transformed.backend), file=sys.stderr)
            print("[amber-mlips] transformed input: {}".format(transformed_path), file=sys.stderr)
            print("[amber-mlips] launcher mode: {}".format(launcher_mode), file=sys.stderr)
            print("[amber-mlips] mm ranks: {}".format(mm_ranks), file=sys.stderr)

        server_launch_prefix = []
        amber_launch_prefix = []
        dvm_cmd_preview = None
        if launcher_mode == "dvm":
            if ns.dry_run:
                dvm_uri_file = "/tmp/amber_mlips_dryrun.dvm.uri"
                dvm_cmd_preview = [
                    prte_bin,
                    "--daemonize",
                    "--report-uri",
                    dvm_uri_file,
                    "--report-pid",
                    "/tmp/amber_mlips_dryrun.dvm.pid",
                ]
                server_launch_prefix = _build_prun_prefix(prun_bin, dvm_uri_file, n_ranks=1)
                amber_launch_prefix = _build_prun_prefix(prun_bin, dvm_uri_file, n_ranks=mm_ranks)
            else:
                dvm_state = _start_dvm(prte_bin, debug=bool(ns.debug), env=subprocess_env)
                server_launch_prefix = _build_prun_prefix(prun_bin, dvm_state["uri_file"], n_ranks=1)
                amber_launch_prefix = _build_prun_prefix(prun_bin, dvm_state["uri_file"], n_ranks=mm_ranks)

        amber_cmd = list(amber_launch_prefix) + [sander_bin] + forward_exec
        if ns.dry_run:
            dry_ready = "/tmp/amber_mlips_dryrun.ready"
            server_base = _build_server_cmd(
                backend=transformed.backend,
                ready_file=dry_ready,
                ml_keywords=transformed.ml_keywords,
                debug=bool(ns.debug),
            )
            server_cmd = list(server_launch_prefix) + server_base
            if dvm_cmd_preview is not None:
                print("[amber-mlips] dvm    cmd: {}".format(" ".join(shlex.quote(x) for x in dvm_cmd_preview)), file=sys.stderr)
            print("[amber-mlips] server cmd: {}".format(" ".join(shlex.quote(x) for x in server_cmd)), file=sys.stderr)
            print("[amber-mlips] amber  cmd: {}".format(" ".join(shlex.quote(x) for x in amber_cmd)), file=sys.stderr)
            return 0

        server_proc, ready_file, server_cmd = _start_server(
            transformed.backend,
            transformed.ml_keywords,
            debug=bool(ns.debug),
            launcher_prefix=server_launch_prefix,
            env=subprocess_env,
        )

        if ns.debug or ns.dry_run:
            if dvm_state is not None:
                print(
                    "[amber-mlips] dvm    cmd: {}".format(
                        " ".join(shlex.quote(x) for x in dvm_state["cmd"])
                    ),
                    file=sys.stderr,
                )
            print("[amber-mlips] server cmd: {}".format(" ".join(shlex.quote(x) for x in server_cmd)), file=sys.stderr)
            print("[amber-mlips] amber  cmd: {}".format(" ".join(shlex.quote(x) for x in amber_cmd)), file=sys.stderr)

        proc = subprocess.run(amber_cmd, env=subprocess_env)
        ret = int(proc.returncode)

        # Give the server a moment to exit gracefully after genmpi finalize.
        if server_proc is not None and server_proc.poll() is None:
            try:
                server_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        return ret

    except (InputTransformError, AmberMLIPSError, OSError, subprocess.SubprocessError) as exc:
        print("[amber-mlips] ERROR: {}".format(exc), file=sys.stderr)
        return 1

    finally:
        _stop_server(server_proc)
        _stop_dvm(dvm_state, env=subprocess_env)
        if ready_file and os.path.exists(ready_file):
            try:
                os.unlink(ready_file)
            except OSError:
                pass
        if generated_input and os.path.exists(generated_input):
            try:
                os.unlink(generated_input)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
