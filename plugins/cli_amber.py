#!/usr/bin/env python3
"""Single-command amber-mlips wrapper (non-MPI qchem backend).

Usage pattern mirrors sander flags:

    amber-mlips -O -i mlmm.in -o mlmm.out -p leap.parm7 -c md.rst7 ...

Internally:
1) transform `mlmm.in` from backend-style qmmm into EXTERN+qc input,
2) stage a private `qchem` shim executable in PATH,
3) run sander/sander.MPI with transformed input,
4) clean up temporary files.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass

from .mdin_transform import InputTransformError, transform_mdin_text


class AmberMLIPSError(RuntimeError):
    """Raised for wrapper-level runtime failures."""


@dataclass(frozen=True)
class InputArgRef:
    """Reference to the original `-i` argument in forwarded sander args."""

    index: int
    inline_path: str
    user_path: str


@dataclass(frozen=True)
class LaunchSpec:
    """Resolved MM launch mode and command prefix."""

    mode: str
    prefix: tuple


def _is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _build_runtime_env():
    """Return subprocess environment with practical OpenMPI defaults."""
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_mca_base_component_show_load_errors", "none")
    env.setdefault("OMPI_MCA_opal_warn_on_missing_libcuda", "0")
    if "OMPI_MCA_opal_cuda_support" not in env and not os.path.exists("/dev/nvidiactl"):
        env["OMPI_MCA_opal_cuda_support"] = "0"

    # Some environments only provide libcudart through conda.
    conda_prefix = env.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        cudart_link = os.path.join(conda_lib, "libcudart.so.12")
        if os.path.isfile(cudart_link):
            ld = env.get("LD_LIBRARY_PATH", "")
            paths = [p for p in ld.split(os.pathsep) if p]
            if conda_lib not in paths:
                env["LD_LIBRARY_PATH"] = conda_lib + (os.pathsep + ld if ld else "")

    return env


def _resolve_user_choice(path_or_name, option_name):
    if not path_or_name:
        return None
    if _is_executable(path_or_name):
        return path_or_name
    found = shutil.which(path_or_name)
    if found:
        return found
    raise AmberMLIPSError(
        "{} not found or not executable: {}".format(option_name, path_or_name)
    )


def _resolve_sander_bin(user_choice, prefer_mpi=False):
    resolved = _resolve_user_choice(user_choice, "--sander-bin")
    if resolved:
        return resolved

    names = ("sander.MPI", "sander") if prefer_mpi else ("sander", "sander.MPI")

    amberhome = os.environ.get("AMBERHOME", "").strip()
    candidates = []
    if amberhome:
        for name in names:
            candidates.append(os.path.join(amberhome, "bin", name))

    fixed = {
        "sander": "/home/apps/amber24/bin/sander",
        "sander.MPI": "/home/apps/amber24/bin/sander.MPI",
    }
    for name in names:
        candidates.append(fixed[name])

    for path in candidates:
        if _is_executable(path):
            return path

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    raise AmberMLIPSError(
        "Could not locate sander executable. Set AMBERHOME or use --sander-bin PATH."
    )


def _resolve_mpi_launcher(user_choice=None):
    resolved = _resolve_user_choice(user_choice, "--mpi-bin")
    if resolved:
        return resolved

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]

    # Prefer non-conda MPI launchers when available, because conda-provided
    # launchers may not match module-loaded AMBER/OpenMPI runtime libraries.
    def iter_candidates(name):
        for entry in path_entries:
            candidate = os.path.join(entry, name)
            if _is_executable(candidate):
                yield candidate

    non_conda = []
    conda = []
    for name in ("mpirun", "mpiexec"):
        for candidate in iter_candidates(name):
            if conda_prefix and os.path.abspath(candidate).startswith(os.path.abspath(conda_prefix) + os.sep):
                conda.append(candidate)
            else:
                non_conda.append(candidate)

    if non_conda:
        return non_conda[0]
    if conda:
        return conda[0]

    raise AmberMLIPSError(
        "MPI launcher not found (mpirun/mpiexec). Set --mpi-bin explicitly."
    )


def _extract_input_arg(argv):
    for i, token in enumerate(argv):
        if token == "-i":
            if i + 1 >= len(argv):
                raise AmberMLIPSError("-i flag was provided without a file path.")
            return InputArgRef(index=i, inline_path=None, user_path=argv[i + 1])

        if token.startswith("-i") and len(token) > 2:
            inline = token[2:]
            return InputArgRef(index=i, inline_path=inline, user_path=inline)

    raise AmberMLIPSError(
        "amber-mlips requires -i <mdin> so the wrapper can transform &qmmm."
    )


def _replace_input_path(argv, input_ref, new_path):
    out = list(argv)
    if input_ref.inline_path is None:
        out[input_ref.index + 1] = new_path
    else:
        out[input_ref.index] = "-i{}".format(new_path)
    return out


def _write_transformed_input(stack, input_path, transformed_text, keep_file):
    src_abs = os.path.abspath(input_path)

    if keep_file:
        out_path = src_abs + ".amber_mlips.qc.in"
        with open(out_path, "w") as handle:
            handle.write(transformed_text)
        return out_path

    fd, tmp_path = tempfile.mkstemp(prefix="amber_mlips_", suffix=".in")
    with os.fdopen(fd, "w") as handle:
        handle.write(transformed_text)
    stack.callback(_safe_unlink, tmp_path)
    return tmp_path


def _safe_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _stage_qchem_shim(stack):
    runtime_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="amber_mlips_qchem_"))
    qchem_path = os.path.join(runtime_dir, "qchem")

    script = (
        "#!/usr/bin/env bash\n"
        "exec {} -m amber_mlips_plugins.nonmpi_qc_shim \"$@\"\n"
    ).format(shlex.quote(sys.executable))

    with open(qchem_path, "w") as handle:
        handle.write(script)
    os.chmod(qchem_path, 0o755)

    return runtime_dir, qchem_path


def _build_launch_spec(mm_ranks, mpi_bin_opt):
    ranks = int(mm_ranks)
    if ranks < 1:
        raise AmberMLIPSError("--mm-ranks must be >= 1.")

    if ranks == 1:
        return LaunchSpec(mode="direct", prefix=tuple())

    mpi_bin = _resolve_mpi_launcher(mpi_bin_opt)
    prefix = [mpi_bin, "-np", str(ranks)]

    # Ensure module/conda runtime env is visible in MPI ranks.
    for key in (
        "PATH",
        "LD_LIBRARY_PATH",
        "AMBER_MLIPS_BACKEND",
        "AMBER_MLIPS_ML_KEYWORDS",
        "AMBER_MLIPS_DEBUG",
    ):
        prefix.extend(["-x", key])

    return LaunchSpec(mode="mpi", prefix=tuple(prefix))


def _build_child_env(base_env, shim_dir, backend, ml_keywords, debug=False):
    env = dict(base_env)
    env["PATH"] = shim_dir + os.pathsep + env.get("PATH", "")
    env["AMBER_MLIPS_BACKEND"] = str(backend)
    env["AMBER_MLIPS_ML_KEYWORDS"] = str(ml_keywords)
    if debug:
        env["AMBER_MLIPS_DEBUG"] = "1"
    else:
        env.pop("AMBER_MLIPS_DEBUG", None)
    return env


def _validate_mm_sander(mm_ranks, sander_bin):
    if int(mm_ranks) <= 1:
        return

    sander_name = os.path.basename(str(sander_bin)).lower()
    if ".mpi" in sander_name:
        return

    raise AmberMLIPSError(
        "--mm-ranks > 1 requires an MPI-capable sander binary (e.g., sander.MPI). "
        "Current binary: {}".format(sander_bin)
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="amber-mlips",
        description="Run AMBER QM/MM with MLIP backend via internal EXTERN+qc bridge.",
        add_help=False,
    )
    parser.add_argument(
        "--sander-bin",
        default=None,
        help="Path to sander executable (default: auto-detect; prefer sander for 1 rank, sander.MPI for >1).",
    )
    parser.add_argument(
        "--mpi-bin",
        default=None,
        help="MPI launcher command/path for MM ranks > 1 (default: auto-detect mpirun/mpiexec).",
    )
    parser.add_argument(
        "--mm-ranks",
        type=int,
        default=1,
        help="MPI ranks for sander (MM side). ML path is always non-MPI qchem shim.",
    )
    parser.add_argument(
        "--keep-transformed-input",
        action="store_true",
        help="Keep transformed mdin as '<input>.amber_mlips.qc.in'.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Transform input and print command, but do not run sander.")
    parser.add_argument("--debug", action="store_true", help="Verbose wrapper/shim logs.")
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


def _print_debug(transformed, transformed_path, launch_spec, mm_ranks, qchem_path):
    print("[amber-mlips] backend: {}".format(transformed.backend), file=sys.stderr)
    print("[amber-mlips] transformed input: {}".format(transformed_path), file=sys.stderr)
    print("[amber-mlips] launcher mode: {}".format(launch_spec.mode), file=sys.stderr)
    print("[amber-mlips] mm ranks: {}".format(int(mm_ranks)), file=sys.stderr)
    print("[amber-mlips] qchem shim: {}".format(qchem_path), file=sys.stderr)
    print(
        "[amber-mlips] ml_keywords: {}".format(
            transformed.ml_keywords if transformed.ml_keywords else "(empty)"
        ),
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

    base_env = _build_runtime_env()

    try:
        with ExitStack() as stack:
            input_ref = _extract_input_arg(forward)

            with open(input_ref.user_path, "r") as handle:
                original_text = handle.read()
            transformed = transform_mdin_text(original_text)

            for warn in transformed.warnings:
                print("[amber-mlips] WARNING: {}".format(warn), file=sys.stderr)

            transformed_path = _write_transformed_input(
                stack=stack,
                input_path=input_ref.user_path,
                transformed_text=transformed.transformed_text,
                keep_file=bool(ns.keep_transformed_input),
            )

            launch_spec = _build_launch_spec(ns.mm_ranks, ns.mpi_bin)
            sander_bin = _resolve_sander_bin(ns.sander_bin, prefer_mpi=(int(ns.mm_ranks) > 1))
            _validate_mm_sander(ns.mm_ranks, sander_bin)

            shim_dir, qchem_path = _stage_qchem_shim(stack)
            child_env = _build_child_env(
                base_env=base_env,
                shim_dir=shim_dir,
                backend=transformed.backend,
                ml_keywords=transformed.ml_keywords,
                debug=bool(ns.debug),
            )

            forward_exec = _replace_input_path(forward, input_ref, transformed_path)
            amber_cmd = list(launch_spec.prefix) + [sander_bin] + forward_exec

            if ns.debug:
                _print_debug(
                    transformed=transformed,
                    transformed_path=transformed_path,
                    launch_spec=launch_spec,
                    mm_ranks=ns.mm_ranks,
                    qchem_path=qchem_path,
                )

            if ns.dry_run:
                print("[amber-mlips] qchem shim: {}".format(qchem_path), file=sys.stderr)
                print(
                    "[amber-mlips] amber  cmd: {}".format(
                        " ".join(shlex.quote(x) for x in amber_cmd)
                    ),
                    file=sys.stderr,
                )
                return 0

            return int(subprocess.run(amber_cmd, env=child_env).returncode)

    except (InputTransformError, AmberMLIPSError, OSError, subprocess.SubprocessError) as exc:
        print("[amber-mlips] ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
