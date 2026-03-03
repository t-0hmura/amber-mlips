"""Auto-detect and patch AMBER EXTERN MPI bug in qm2_extern_module.F90.

AMBER 24 (and earlier) has a bug where non-rank-0 MPI tasks return from
qm2_extern_get_qm_forces() without initializing output arrays (escf,
dxyzqm, dxyzcl).  The subsequent force collation in qm_mm.F90 runs on
ALL ranks, adding uninitialised garbage to the force array.

The fix is a 3-line addition: zero-initialize the outputs before the
early return.  This module detects the bug in the installed AMBER source,
builds a patched sander.MPI, and caches it under ~/.amber-mlips/.
"""

from __future__ import absolute_import, division, print_function

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Bug detection
# ---------------------------------------------------------------------------

_SANDER_SUBDIR = os.path.join("AmberTools", "src", "sander")
_BUG_SOURCE = os.path.join(_SANDER_SUBDIR, "qm2_extern_module.F90")
_FLAGS_MAKE = os.path.join(
    _SANDER_SUBDIR,
    "CMakeFiles",
    "sander_base_obj_mpi.dir",
    "flags.make",
)
_LINK_TXT = os.path.join(
    _SANDER_SUBDIR,
    "CMakeFiles",
    "sander.MPI.dir",
    "link.txt",
)
_OBJ_PATTERN = "CMakeFiles/sander_base_obj_mpi.dir/qm2_extern_module.F90.o"

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".amber-mlips")
_CACHED_BIN = os.path.join(_CACHE_DIR, "sander_patched.MPI")
_CACHED_META = os.path.join(_CACHE_DIR, "sander_patched.meta")

# The 3-line fix: zero-initialise outputs on non-rank-0 before early return.
_BUG_PATTERN = re.compile(
    r"^\s*if\s*\(\s*mytaskid\s*/=\s*0\s*\)\s*return\s*$", re.MULTILINE
)
_FIX_MARKER = "escf   = 0.0d0"

_PATCHED_BLOCK = """\
    if ( mytaskid /= 0 ) then
      escf   = 0.0d0
      dxyzqm = 0.0d0
      if (nclatoms > 0) dxyzcl = 0.0d0
      return
    end if"""


def find_amber_source(sander_bin=None):
    """Locate the AMBER source tree (containing AmberTools/src/sander/).

    Returns the root of the source tree, or *None* if not found.
    """
    candidates = []

    amberhome = os.environ.get("AMBERHOME", "").strip()
    if amberhome:
        # Common layouts: $AMBERHOME_src, $AMBERHOME/../amber*_src, in-tree
        parent = os.path.dirname(amberhome)
        base = os.path.basename(amberhome)
        candidates.append(amberhome + "_src")
        candidates.append(os.path.join(parent, base + "_src"))
        # In-tree build: source is directly under AMBERHOME
        candidates.append(amberhome)
        # Search siblings that look like amber source dirs
        if os.path.isdir(parent):
            for name in sorted(os.listdir(parent)):
                if "amber" in name.lower() and "src" in name.lower():
                    candidates.append(os.path.join(parent, name))

    if sander_bin:
        # /path/to/amber24/bin/sander.MPI  →  /path/to/amber24
        real = os.path.realpath(sander_bin)
        bindir = os.path.dirname(real)
        amber_root = os.path.dirname(bindir)
        parent = os.path.dirname(amber_root)
        base = os.path.basename(amber_root)
        candidates.append(amber_root + "_src")
        candidates.append(os.path.join(parent, base + "_src"))
        candidates.append(amber_root)

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, _BUG_SOURCE)):
            return cand
    return None


def find_amber_build(source_root):
    """Locate the CMake build directory for sander MPI.

    Returns the build root (parent of AmberTools/), or *None*.
    """
    # Common: $SOURCE_ROOT/build
    for sub in ("build", ""):
        build = os.path.join(source_root, sub) if sub else source_root
        if os.path.isfile(os.path.join(build, _FLAGS_MAKE)):
            return build
    return None


def has_extern_mpi_bug(source_root):
    """Return True if the AMBER source has the uninitialised-output bug."""
    path = os.path.join(source_root, _BUG_SOURCE)
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError:
        return False
    return bool(_BUG_PATTERN.search(text)) and (_FIX_MARKER not in text)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _source_mtime(source_root):
    path = os.path.join(source_root, _BUG_SOURCE)
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def get_cached_sander(source_root):
    """Return cached patched sander path if valid, else None."""
    if not os.path.isfile(_CACHED_BIN):
        return None
    if not os.path.isfile(_CACHED_META):
        return None
    try:
        with open(_CACHED_META, "r") as f:
            meta = json.load(f)
        if meta.get("source_mtime", 0) >= _source_mtime(source_root):
            return _CACHED_BIN
    except (OSError, ValueError, KeyError):
        pass
    return None


def _save_meta(source_root):
    meta = {
        "source_root": source_root,
        "source_mtime": _source_mtime(source_root),
    }
    with open(_CACHED_META, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Patch & build
# ---------------------------------------------------------------------------

def _apply_patch(src_path, dst_path):
    """Copy source, replace the buggy early-return with the fix."""
    with open(src_path, "r") as f:
        text = f.read()
    patched = _BUG_PATTERN.sub(_PATCHED_BLOCK, text)
    if patched == text:
        raise RuntimeError("Could not locate bug pattern in {}".format(src_path))
    with open(dst_path, "w") as f:
        f.write(patched)


def _parse_flags_make(build_root):
    """Extract compiler, defines, includes, and flags from CMake flags.make."""
    path = os.path.join(build_root, _FLAGS_MAKE)
    with open(path, "r") as f:
        text = f.read()

    def _get(key):
        m = re.search(r"^{}\s*=\s*(.*)$".format(re.escape(key)), text, re.MULTILINE)
        return m.group(1).strip() if m else ""

    # Extract Fortran compiler from the comment line
    fc_match = re.search(r"compile Fortran with\s+(\S+)", text)
    fc = fc_match.group(1) if fc_match else "gfortran"

    return {
        "fc": fc,
        "defines": _get("Fortran_DEFINES"),
        "includes": _get("Fortran_INCLUDES"),
        "flags": _get("Fortran_FLAGS"),
    }


def build_patched_sander(source_root, build_root):
    """Build a patched sander.MPI and cache it under ~/.amber-mlips/."""
    os.makedirs(_CACHE_DIR, exist_ok=True)

    src_file = os.path.join(source_root, _BUG_SOURCE)
    cfg = _parse_flags_make(build_root)

    sander_build = os.path.join(build_root, _SANDER_SUBDIR)
    link_txt_path = os.path.join(build_root, _LINK_TXT)

    with tempfile.TemporaryDirectory(prefix="amber_mlips_patch_") as tmpdir:
        # 1. Apply patch
        patched_src = os.path.join(tmpdir, "qm2_extern_module.F90")
        _apply_patch(src_file, patched_src)

        # 2. Compile patched object
        patched_obj = os.path.join(tmpdir, "qm2_extern_module.F90.o")
        moddir = os.path.join(tmpdir, "moddir")
        os.makedirs(moddir, exist_ok=True)

        # The include path must contain sander source dir for #include "parallel.h" etc.
        sander_src = os.path.join(source_root, _SANDER_SUBDIR)
        extra_includes = "-I{}".format(sander_src)

        # Replace the original -J (module output dir) with our temp dir.
        flags = re.sub(r"-J\S+", "", cfg["flags"]).strip()

        compile_cmd = (
            "{fc} {defines} {includes} {extra_inc} {flags} "
            "-J{moddir} -O0 -c {src} -o {obj}"
        ).format(
            fc=cfg["fc"],
            defines=cfg["defines"],
            includes=cfg["includes"],
            extra_inc=extra_includes,
            flags=flags,
            moddir=moddir,
            src=patched_src,
            obj=patched_obj,
        )

        _run_cmd(compile_cmd, "compile", cwd=tmpdir)

        # 3. Re-link sander.MPI with patched object
        with open(link_txt_path, "r") as f:
            link_cmd = f.read().strip()

        link_cmd = link_cmd.replace(_OBJ_PATTERN, patched_obj)
        link_cmd = re.sub(r"-o\s+sander\.MPI", "-o " + _CACHED_BIN, link_cmd)

        # Fix missing libopenblas.so symlink if needed
        if "/usr/lib64/libopenblas.so" in link_cmd:
            real_lib = "/usr/lib64/libopenblas.so"
            if not os.path.exists(real_lib):
                for candidate in ("/usr/lib64/libopenblas.so.0",):
                    if os.path.exists(candidate):
                        local_link = os.path.join(tmpdir, "libopenblas.so")
                        os.symlink(candidate, local_link)
                        link_cmd = link_cmd.replace(real_lib, local_link)
                        break

        _run_cmd(link_cmd, "link", cwd=sander_build)

    _save_meta(source_root)
    return _CACHED_BIN


def _run_cmd(cmd, stage, cwd=None):
    """Run a shell command, raising on failure."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "sander patch {stage} failed (rc={rc}).\n"
            "Command: {cmd}\n"
            "stderr:\n{err}".format(
                stage=stage,
                rc=result.returncode,
                cmd=cmd,
                err=result.stderr.decode("utf-8", errors="replace"),
            )
        )


# ---------------------------------------------------------------------------
# High-level entry point (called from cli_amber.py)
# ---------------------------------------------------------------------------

def ensure_patched_sander(sander_bin):
    """Check for the EXTERN MPI bug and return a patched sander.MPI path.

    Returns *sander_bin* unchanged if:
    - the bug is not present (already fixed upstream), or
    - the user already specified a custom --sander-bin.

    Raises AmberMLIPSError-compatible RuntimeError if patching fails.
    """
    source_root = find_amber_source(sander_bin)
    if source_root is None:
        raise RuntimeError(
            "AMBER source directory not found. Cannot verify/patch sander for "
            "multi-rank MPI.\n"
            "Options:\n"
            "  1. Set AMBERHOME to your AMBER installation directory.\n"
            "  2. Use --mm-ranks 1 (no MPI, no bug).\n"
            "  3. Use --sander-bin /path/to/patched/sander.MPI."
        )

    if not has_extern_mpi_bug(source_root):
        # Source is already fixed (future AMBER version). No patch needed.
        return sander_bin

    # Bug detected. Check cache.
    cached = get_cached_sander(source_root)
    if cached:
        print(
            "[amber-mlips] Using cached patched sander.MPI: {}".format(cached),
            file=sys.stderr, flush=True,
        )
        return cached

    # Build patched sander.
    build_root = find_amber_build(source_root)
    if build_root is None:
        raise RuntimeError(
            "AMBER build directory not found under {}.\n"
            "Cannot build patched sander.MPI automatically.\n"
            "Options:\n"
            "  1. Use --mm-ranks 1 (no MPI, no bug).\n"
            "  2. Use --sander-bin /path/to/patched/sander.MPI.".format(
                source_root
            )
        )

    print(
        "[amber-mlips] AMBER EXTERN MPI bug detected in {}.\n"
        "[amber-mlips] Building patched sander.MPI (one-time)...".format(
            os.path.join(source_root, _BUG_SOURCE),
        ),
        file=sys.stderr, flush=True,
    )

    patched = build_patched_sander(source_root, build_root)

    print(
        "[amber-mlips] Patched sander.MPI built and cached: {}".format(patched),
        file=sys.stderr, flush=True,
    )
    return patched
