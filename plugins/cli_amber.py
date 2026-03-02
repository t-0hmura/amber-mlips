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


def _resolve_sander_bin(user_choice, prefer_mpi=False):
    if user_choice:
        if os.path.isfile(user_choice) and os.access(user_choice, os.X_OK):
            return user_choice
        found = shutil.which(user_choice)
        if found:
            return found
        raise AmberMLIPSError("--sander-bin not found or not executable: {}".format(user_choice))

    if prefer_mpi:
        names = ("sander.MPI", "sander")
    else:
        names = ("sander", "sander.MPI")

    candidates = []
    amberhome = os.environ.get("AMBERHOME", "").strip()
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
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    raise AmberMLIPSError(
        "Could not locate sander executable. Set AMBERHOME or use --sander-bin PATH."
    )


def _resolve_mpi_launcher(user_choice=None):
    if user_choice:
        if os.path.isfile(user_choice) and os.access(user_choice, os.X_OK):
            return user_choice
        found = shutil.which(user_choice)
        if found:
            return found
        raise AmberMLIPSError("--mpi-bin not found or not executable: {}".format(user_choice))

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]

    # Prefer non-conda MPI launchers when available, because conda-provided
    # launchers may not match module-loaded AMBER/OpenMPI runtime libraries.
    def _iter_paths(name):
        for entry in path_entries:
            cand = os.path.join(entry, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                yield cand

    non_conda = []
    conda = []
    for name in ("mpirun", "mpiexec"):
        for cand in _iter_paths(name):
            if conda_prefix and os.path.abspath(cand).startswith(os.path.abspath(conda_prefix) + os.sep):
                conda.append(cand)
            else:
                non_conda.append(cand)

    if non_conda:
        return non_conda[0]
    if conda:
        return conda[0]

    raise AmberMLIPSError(
        "MPI launcher not found (mpirun/mpiexec). Set --mpi-bin explicitly."
    )


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


def _write_transformed_input(input_path, transformed_text, keep_file):
    src_abs = os.path.abspath(input_path)

    if keep_file:
        out_path = src_abs + ".amber_mlips.qc.in"
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


def _stage_qchem_shim():
    runtime_dir = tempfile.mkdtemp(prefix="amber_mlips_qchem_")
    qchem_path = os.path.join(runtime_dir, "qchem")

    script = (
        "#!/usr/bin/env bash\n"
        "exec {} -m amber_mlips_plugins.nonmpi_qc_shim \"$@\"\n"
    ).format(shlex.quote(sys.executable))

    with open(qchem_path, "w") as handle:
        handle.write(script)

    os.chmod(qchem_path, 0o755)
    return runtime_dir, qchem_path


def _build_amber_launch(mode, mm_ranks, mpi_bin_opt):
    launch_mode = str(mode or "auto").strip().lower()
    if launch_mode not in {"auto", "mpi", "direct", "dvm"}:
        raise AmberMLIPSError("Unknown --launcher-mode '{}'.".format(mode))

    warnings = []
    if launch_mode == "dvm":
        warnings.append("--launcher-mode dvm is deprecated in non-MPI mode; using mpi launcher semantics.")
        launch_mode = "mpi"

    ranks = int(mm_ranks)
    if ranks < 1:
        raise AmberMLIPSError("--mm-ranks must be >= 1.")

    if ranks == 1 and launch_mode == "auto":
        return "direct", [], warnings

    if launch_mode == "direct":
        if ranks > 1:
            raise AmberMLIPSError("--mm-ranks > 1 requires MPI launch (use --launcher-mode auto or mpi).")
        return "direct", [], warnings

    mpi_bin = _resolve_mpi_launcher(mpi_bin_opt)
    nprocs = ranks if ranks > 1 else 1
    prefix = [mpi_bin, "-np", str(nprocs)]
    # Ensure module/conda runtime env is visible in MPI ranks.
    for key in (
        "PATH",
        "LD_LIBRARY_PATH",
        "AMBER_MLIPS_BACKEND",
        "AMBER_MLIPS_ML_KEYWORDS",
        "AMBER_MLIPS_DEBUG",
    ):
        prefix.extend(["-x", key])
    return "mpi", prefix, warnings


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
        "--launcher-mode",
        default="auto",
        choices=("auto", "mpi", "direct", "dvm"),
        help="MM launcher: auto(direct for 1 rank, mpi for >1), mpi(force mpi launcher), direct(no mpi launcher).",
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

    generated_input = None
    transformed_path = None
    shim_dir = None
    qchem_path = None
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

        mm_ranks = int(ns.mm_ranks)
        if mm_ranks < 1:
            raise AmberMLIPSError("--mm-ranks must be >= 1.")

        launch_mode, amber_prefix, mode_warnings = _build_amber_launch(
            ns.launcher_mode,
            mm_ranks,
            ns.mpi_bin,
        )
        for warn in mode_warnings:
            print("[amber-mlips] WARNING: {}".format(warn), file=sys.stderr)

        sander_bin = _resolve_sander_bin(ns.sander_bin, prefer_mpi=(mm_ranks > 1))
        if mm_ranks > 1:
            sander_name = os.path.basename(str(sander_bin)).lower()
            if ".mpi" not in sander_name:
                raise AmberMLIPSError(
                    "--mm-ranks > 1 requires an MPI-capable sander binary (e.g., sander.MPI). "
                    "Current binary: {}".format(sander_bin)
                )

        shim_dir, qchem_path = _stage_qchem_shim()
        subprocess_env["PATH"] = shim_dir + os.pathsep + subprocess_env.get("PATH", "")
        subprocess_env["AMBER_MLIPS_BACKEND"] = str(transformed.backend)
        subprocess_env["AMBER_MLIPS_ML_KEYWORDS"] = str(transformed.ml_keywords)
        if ns.debug:
            subprocess_env["AMBER_MLIPS_DEBUG"] = "1"

        forward_exec = _replace_input_path(forward, i_idx, inline_i, transformed_path)
        amber_cmd = list(amber_prefix) + [sander_bin] + forward_exec

        if ns.debug:
            print("[amber-mlips] backend: {}".format(transformed.backend), file=sys.stderr)
            print("[amber-mlips] transformed input: {}".format(transformed_path), file=sys.stderr)
            print("[amber-mlips] launcher mode: {}".format(launch_mode), file=sys.stderr)
            print("[amber-mlips] mm ranks: {}".format(mm_ranks), file=sys.stderr)
            print("[amber-mlips] qchem shim: {}".format(qchem_path), file=sys.stderr)
            print(
                "[amber-mlips] ml_keywords: {}".format(
                    transformed.ml_keywords if transformed.ml_keywords else "(empty)"
                ),
                file=sys.stderr,
            )

        if ns.dry_run:
            print("[amber-mlips] qchem shim: {}".format(qchem_path), file=sys.stderr)
            print("[amber-mlips] amber  cmd: {}".format(" ".join(shlex.quote(x) for x in amber_cmd)), file=sys.stderr)
            return 0

        proc = subprocess.run(amber_cmd, env=subprocess_env)
        return int(proc.returncode)

    except (InputTransformError, AmberMLIPSError, OSError, subprocess.SubprocessError) as exc:
        print("[amber-mlips] ERROR: {}".format(exc), file=sys.stderr)
        return 1

    finally:
        if generated_input and os.path.exists(generated_input):
            try:
                os.unlink(generated_input)
            except OSError:
                pass
        if shim_dir and os.path.isdir(shim_dir):
            try:
                shutil.rmtree(shim_dir)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
