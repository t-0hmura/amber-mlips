#!/usr/bin/env python3
"""Transform AMBER mdin for amber-mlips single-command workflow.

User-facing input (in &qmmm):
- qm_theory = "uma"|"orb"|"mace"|"aimnet2"
- ml_keywords = "..."
- mlcut = <float>   (optional; mapped to qmcut)

Internal transformed input:
- qm_theory = 'EXTERN'
- ml_keywords / mlcut removed
- qmcut set from mlcut if provided
- &genmpi block injected

Notes:
- AMBER EXTERN path requires qm_ewald=0 and qmgb=0.
"""

from __future__ import absolute_import, division, print_function

import re
from dataclasses import dataclass


SUPPORTED_BACKENDS = {"uma", "orb", "mace", "aimnet2"}


class InputTransformError(RuntimeError):
    """Raised when the user mdin cannot be transformed safely."""


@dataclass
class TransformResult:
    backend: str
    ml_keywords: str
    transformed_text: str
    warnings: list


def _unquote(value):
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _split_namelist_entries(body):
    """Split namelist body into top-level comma-separated entries.

    Commas inside quotes are preserved. Inline comments beginning with '!'
    are stripped before tokenization.
    """
    out = []
    buf = []
    quote = None
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]

        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "!":
            # Skip inline comment until newline.
            while i < n and body[i] != "\n":
                i += 1
            continue

        if ch == ",":
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    token = "".join(buf).strip()
    if token:
        out.append(token)

    return out


def _parse_namelist_entries(body):
    """Parse tokens into ordered key/value map.

    Returns:
      order: list[str]  (first-seen key order)
      values: dict[str, str]  (last assignment wins)
      key_style: dict[str, str]  (original key spelling for output)
    """
    order = []
    values = {}
    key_style = {}

    for token in _split_namelist_entries(body):
        if "=" not in token:
            continue
        left, right = token.split("=", 1)
        key_raw = left.strip()
        key = key_raw.lower()
        val = right.strip()

        if key not in key_style:
            key_style[key] = key_raw
            order.append(key)

        values[key] = val

    return order, values, key_style


def _is_zeroish(value):
    text = _unquote(value).strip().lower().replace("d", "e")
    try:
        return abs(float(text)) < 1.0e-12
    except Exception:
        return text in {"0", "0.0", "+0", "-0"}


_QMMM_RE = re.compile(
    r"(?ims)^[ \t]*&qmmm\b(?P<body>.*?)^[ \t]*/[ \t]*$",
    re.MULTILINE,
)

_GENMPI_RE = re.compile(
    r"(?ims)^[ \t]*&genmpi\b.*?^[ \t]*/[ \t]*\n?",
    re.MULTILINE,
)


def _build_qmmm_block(order, values, key_style):
    lines = [" &qmmm"]
    for key in order:
        if key not in values:
            continue
        out_key = key_style.get(key, key)
        out_val = values[key]
        lines.append("  {}={},".format(out_key, out_val))
    lines.append(" /")
    return "\n".join(lines) + "\n"


def _build_genmpi_block(backend):
    method = {
        "uma": "UMA",
        "orb": "ORB",
        "mace": "MACE",
        "aimnet2": "AIMNET2",
    }[backend]

    return (
        "&genmpi\n"
        "  method='{}',\n"
        "  basis='MLIP',\n"
        "  ntpr=1,\n"
        "  debug=0,\n"
        "  dipole=0,\n"
        "/\n"
    ).format(method)


def transform_mdin_text(text):
    match = _QMMM_RE.search(text)
    if not match:
        raise InputTransformError("&qmmm namelist was not found in input mdin.")

    order, values, key_style = _parse_namelist_entries(match.group("body"))
    warnings = []

    qm_theory_value = values.get("qm_theory")
    if qm_theory_value is None:
        raise InputTransformError("&qmmm must contain qm_theory.")

    backend = _unquote(qm_theory_value).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise InputTransformError(
            "qm_theory='{}' is not supported by amber-mlips. Use one of: {}".format(
                backend,
                ", ".join(sorted(SUPPORTED_BACKENDS)),
            )
        )

    ml_keywords = _unquote(values.get("ml_keywords", "")).strip()

    # Remove plugin-only keys from qmmm before passing to AMBER.
    if "ml_keywords" in values:
        del values["ml_keywords"]
    if "ml_keywords" in order:
        order.remove("ml_keywords")

    if "mlcut" in values:
        values["qmcut"] = values["mlcut"]
        if "qmcut" not in order:
            order.append("qmcut")
        del values["mlcut"]
        if "mlcut" in order:
            order.remove("mlcut")

    # Force EXTERN + allowed EXTERN settings.
    values["qm_theory"] = "'EXTERN'"
    if "qm_theory" not in order:
        order.append("qm_theory")

    if "qm_ewald" in values and not _is_zeroish(values["qm_ewald"]):
        warnings.append("qm_ewald was set to 0 because qm_theory=EXTERN requires qm_ewald=0 in AMBER.")
    values["qm_ewald"] = "0"
    if "qm_ewald" not in order:
        order.append("qm_ewald")

    if "qmgb" in values and not _is_zeroish(values["qmgb"]):
        warnings.append("qmgb was set to 0 because qm_theory=EXTERN requires qmgb=0 in AMBER.")
    values["qmgb"] = "0"
    if "qmgb" not in order:
        order.append("qmgb")

    # Remove deprecated alias field if present.
    if "qmtheory" in values:
        del values["qmtheory"]
    if "qmtheory" in order:
        order.remove("qmtheory")

    new_qmmm = _build_qmmm_block(order, values, key_style)

    rewritten = text[: match.start()] + new_qmmm + text[match.end() :]
    rewritten = _GENMPI_RE.sub("", rewritten)

    if not rewritten.endswith("\n"):
        rewritten += "\n"
    rewritten += "\n" + _build_genmpi_block(backend)

    return TransformResult(
        backend=backend,
        ml_keywords=ml_keywords,
        transformed_text=rewritten,
        warnings=warnings,
    )
