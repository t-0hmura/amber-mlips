#!/usr/bin/env python3
"""Persistent MLIP model server and thin client.

The server loads the MLIP model once and keeps it resident in memory,
accepting evaluation requests over a Unix domain socket.  The client
is a lightweight function called by the AMBER runner each time
the external program is invoked, avoiding repeated model loading.

Wire protocol
-------------
Each message is a length-prefixed JSON blob:

    [4-byte big-endian uint32: payload length] [UTF-8 JSON payload]

Actions
-------
- ``evaluate`` : run evaluator.evaluate() and return results
- ``ping``     : health check, returns ``{"status": "ok"}``
- ``shutdown`` : graceful server stop
"""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


class ServerError(RuntimeError):
    """Raised when server communication fails."""


def _pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Wire protocol helpers
# ---------------------------------------------------------------------------

_HEADER_FMT = "!I"  # 4-byte big-endian unsigned int
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MSG_SIZE = 256 * 1024 * 1024  # 256 MB safety cap
_BIN_MAGIC = 0x01  # first payload byte for binary format


def _recv_exact(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# JSON wire helpers (legacy, used by C shim and small messages)
# ---------------------------------------------------------------------------

def _send_msg(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(data))
    sock.sendall(header + data)


def _recv_msg(sock):
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        return None
    (length,) = struct.unpack(_HEADER_FMT, header)
    if length > _MAX_MSG_SIZE:
        raise ServerError("Message too large: {} bytes".format(length))
    data = _recv_exact(sock, length)
    if data is None:
        raise ServerError("Connection closed mid-message")
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# Binary wire helpers (fast path for large numeric arrays)
#
# Format:  [4-byte total length][\x01][4-byte json_len][json_meta][array data]
#
# json_meta contains "_bin" key mapping array names to [shape...] lists.
# Array data follows in declaration order as contiguous float64 bytes.
# Auto-detected by first payload byte: '{' → JSON, \x01 → binary.
# ---------------------------------------------------------------------------

def _send_msg_bin(sock, meta, arrays=None):
    """Send a message using the binary wire format.

    *meta* is a JSON-serializable dict (small metadata).
    *arrays* is an OrderedDict/list-of-tuples of ``(name, ndarray_or_None)``.
    """
    if arrays is None:
        arrays = []
    bin_desc = {}
    buf_parts = []
    for name, arr in (arrays.items() if hasattr(arrays, "items") else arrays):
        if arr is None:
            continue
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        bin_desc[name] = list(arr.shape)
        buf_parts.append(arr.tobytes())

    meta_copy = dict(meta)
    if bin_desc:
        meta_copy["_bin"] = bin_desc

    meta_bytes = json.dumps(meta_copy).encode("utf-8")
    array_bytes = b"".join(buf_parts)

    # \x01 magic + json_len(4) + json + array data
    payload = (
        bytes([_BIN_MAGIC])
        + struct.pack(_HEADER_FMT, len(meta_bytes))
        + meta_bytes
        + array_bytes
    )
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def _recv_msg_auto(sock):
    """Receive a message, auto-detecting JSON or binary format.

    Returns ``(obj_dict, format_flag)`` where *format_flag* is
    ``'json'`` or ``'bin'``.  For binary messages, numpy arrays listed
    in ``obj["_bin"]`` are restored as ``np.float64`` arrays in *obj*.
    """
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        return None, "json"
    (length,) = struct.unpack(_HEADER_FMT, header)
    if length > _MAX_MSG_SIZE:
        raise ServerError("Message too large: {} bytes".format(length))
    data = _recv_exact(sock, length)
    if data is None:
        raise ServerError("Connection closed mid-message")

    if len(data) == 0:
        raise ServerError("Empty payload")

    # Auto-detect format by first byte.
    if data[0] == _BIN_MAGIC:
        if len(data) < 5:
            raise ServerError("Binary message too short")
        (json_len,) = struct.unpack(_HEADER_FMT, data[1:5])
        meta = json.loads(data[5:5 + json_len].decode("utf-8"))
        offset = 5 + json_len
        bin_desc = meta.pop("_bin", {})
        for name, shape in bin_desc.items():
            nbytes = int(np.prod(shape)) * 8  # float64 = 8 bytes
            arr_data = data[offset:offset + nbytes]
            meta[name] = np.frombuffer(arr_data, dtype=np.float64).reshape(shape)
            offset += nbytes
        return meta, "bin"
    else:
        obj = json.loads(data.decode("utf-8"))
        return obj, "json"


def _send_msg_auto(sock, obj, fmt="json"):
    """Send response matching the client's format."""
    if fmt == "bin":
        # Separate numpy arrays from the dict.
        meta = {}
        arrays = []
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                arrays.append((k, v))
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, float, int)):
                # Heuristic: large nested lists that were numpy arrays.
                try:
                    arr = np.asarray(v, dtype=np.float64)
                    if arr.size > 20:
                        arrays.append((k, arr))
                        continue
                except (ValueError, TypeError):
                    pass
                meta[k] = v
            else:
                meta[k] = v
        _send_msg_bin(sock, meta, arrays)
    else:
        # JSON path: convert numpy arrays to lists for json.dumps().
        json_obj = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                json_obj[k] = v.tolist()
            else:
                json_obj[k] = v
        _send_msg(sock, json_obj)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class MLIPServer(object):
    """Single-threaded Unix domain socket server wrapping an evaluator."""

    def __init__(self, evaluator, socket_path, idle_timeout=600, parent_pid=None,
                 embedcharge_opts=None):
        self.evaluator = evaluator
        self.socket_path = os.path.abspath(socket_path)
        self.idle_timeout = float(idle_timeout)
        self.parent_pid = int(parent_pid) if parent_pid is not None else None
        self._running = False
        self._last_activity = time.time()
        # embedcharge_opts: dict with keys xtb_cmd, xtb_acc, xtb_workdir,
        # xtb_keep_files, ncores — or None if embedcharge is disabled.
        self._embedcharge_opts = dict(embedcharge_opts) if embedcharge_opts else None

        # Persistent xTB worker pool for embedcharge correction.
        self._embedcharge_pool = None
        if self._embedcharge_opts:
            try:
                from .xtb_embedcharge_correction import EmbedchargeWorkerPool
            except ImportError:
                from xtb_embedcharge_correction import EmbedchargeWorkerPool
            opts = self._embedcharge_opts
            self._embedcharge_pool = EmbedchargeWorkerPool(
                xtb_cmd=str(opts.get("xtb_cmd", "xtb")),
                xtb_acc=float(opts.get("xtb_acc", 0.2)),
                xtb_workdir=str(opts.get("xtb_workdir", "tmp")),
                xtb_keep_files=bool(opts.get("xtb_keep_files", False)),
                ncores=int(opts.get("ncores", 4)),
            )

    def serve_forever(self):
        if os.path.exists(self.socket_path):
            if server_is_alive(self.socket_path, timeout=2.0):
                print(
                    "[mlip-server] Another server is already running at {}".format(
                        self.socket_path
                    ),
                    file=sys.stderr,
                )
                return
            os.unlink(self.socket_path)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        srv.listen(5)
        srv.settimeout(1.0)

        self._running = True
        self._last_activity = time.time()

        prev_sigint = None
        prev_sigterm = None
        try:
            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, OSError):
            pass  # Not in main thread; skip signal handling

        print(
            "[mlip-server] Listening on {} (idle_timeout={}s, evaluator={}, parent_pid={})".format(
                self.socket_path,
                int(self.idle_timeout),
                self.evaluator.__class__.__name__,
                self.parent_pid if self.parent_pid is not None else "none",
            ),
            file=sys.stderr,
            flush=True,
        )

        try:
            while self._running:
                if self.parent_pid is not None and not _pid_is_alive(self.parent_pid):
                    print(
                        "[mlip-server] Parent process {} exited, shutting down.".format(
                            self.parent_pid
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                # Idle timeout check
                if time.time() - self._last_activity > self.idle_timeout:
                    print(
                        "[mlip-server] Idle timeout reached, shutting down.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue

                try:
                    self._handle_connection(conn)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                finally:
                    conn.close()
        finally:
            srv.close()
            if self._embedcharge_pool is not None:
                try:
                    self._embedcharge_pool.shutdown()
                except Exception:
                    pass
            if os.path.exists(self.socket_path):
                try:
                    os.unlink(self.socket_path)
                except OSError:
                    pass
            if prev_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass
            print("[mlip-server] Shut down.", file=sys.stderr, flush=True)

    def _handle_signal(self, signum, frame):
        print(
            "\n[mlip-server] Signal {} received, shutting down...".format(signum),
            file=sys.stderr,
            flush=True,
        )
        self._running = False

    def _handle_connection(self, conn):
        # Support persistent connections: loop reading requests until the
        # client disconnects or a short inter-message timeout fires.
        conn.settimeout(600.0)
        while self._running:
            try:
                request, client_fmt = _recv_msg_auto(conn)
            except socket.timeout:
                break
            except Exception:
                break
            if request is None:
                break  # Client disconnected

            self._last_activity = time.time()
            action = request.get("action", "")

            if action == "ping":
                _send_msg_auto(conn, {"status": "ok", "message": "pong"}, client_fmt)
            elif action == "shutdown":
                _send_msg_auto(conn, {"status": "ok", "message": "shutting down"}, client_fmt)
                self._running = False
                return
            elif action == "evaluate":
                response = self._do_evaluate(request)
                _send_msg_auto(conn, response, client_fmt)
            else:
                _send_msg_auto(
                    conn, {"status": "error", "message": "Unknown action: {}".format(action)},
                    client_fmt,
                )

            # Short timeout for next message within same connection.
            # If no new request arrives within 2s, close and go back to accept.
            conn.settimeout(2.0)

    def _do_evaluate(self, request):
        try:
            symbols = request["symbols"]
            coords_ang = np.asarray(request["coords_ang"], dtype=np.float64)
            charge = int(request["charge"])
            multiplicity = int(request["multiplicity"])
            need_forces = bool(request.get("need_forces", True))
            need_hessian = bool(request.get("need_hessian", False))
            hessian_mode = str(request.get("hessian_mode", "Analytical"))
            hessian_step = float(request.get("hessian_step", 1.0e-3))

            energy_ev, forces_ev_ang, hess_ev_ang2 = self.evaluator.evaluate(
                symbols=symbols,
                coords_ang=coords_ang,
                charge=charge,
                multiplicity=multiplicity,
                need_forces=need_forces,
                need_hessian=need_hessian,
                hessian_mode=hessian_mode,
                hessian_step=hessian_step,
            )

            # Apply embedcharge correction if MM data is present and enabled.
            mm_coords_raw = request.get("mm_coords_ang")
            mm_charges_raw = request.get("mm_charges")
            forces_mm_ev_ang = None

            if (self._embedcharge_opts and mm_coords_raw is not None
                    and mm_charges_raw is not None):
                mm_coords = np.asarray(mm_coords_raw, dtype=np.float64).reshape(-1, 3)
                mm_charges = np.asarray(mm_charges_raw, dtype=np.float64).reshape(-1)
                ncl = int(mm_coords.shape[0])
                if ncl > 0:
                    opts = self._embedcharge_opts
                    nq = int(coords_ang.reshape(-1, 3).shape[0])

                    if self._embedcharge_pool is not None:
                        # Use persistent worker pool (fast path).
                        from .xtb_embedcharge_correction import _assemble_full_force
                        de, df_q, df_m = self._embedcharge_pool.delta_embed_minus_vac(
                            symbols=symbols,
                            coords_q_ang=coords_ang,
                            mm_coords_ang=mm_coords,
                            mm_charges=mm_charges,
                            charge=charge,
                            multiplicity=multiplicity,
                            need_forces=need_forces,
                            ncores=int(opts.get("ncores", 4)),
                        )
                        if need_forces and df_q is not None:
                            df_full = _assemble_full_force(df_q, df_m)
                        else:
                            df_full = None
                    else:
                        # Fallback to non-persistent path.
                        from .xtb_embedcharge_correction import delta_embedcharge_minus_noembed
                        de, df_full, _ = delta_embedcharge_minus_noembed(
                            symbols=symbols,
                            coords_q_ang=coords_ang,
                            mm_coords_ang=mm_coords,
                            mm_charges=mm_charges,
                            charge=charge,
                            multiplicity=multiplicity,
                            need_forces=need_forces,
                            need_hessian=False,
                            xtb_cmd=str(opts.get("xtb_cmd", "xtb")),
                            xtb_acc=float(opts.get("xtb_acc", 0.2)),
                            xtb_workdir=str(opts.get("xtb_workdir", "tmp")),
                            xtb_keep_files=bool(opts.get("xtb_keep_files", False)),
                            ncores=int(opts.get("ncores", 4)),
                            hessian_step=hessian_step,
                        )

                    energy_ev = float(energy_ev) + float(de)
                    if df_full is not None:
                        df_full = np.asarray(df_full, dtype=np.float64).reshape(nq + ncl, 3)
                        if forces_ev_ang is not None:
                            forces_ev_ang = np.asarray(forces_ev_ang, dtype=np.float64).reshape(nq, 3)
                            forces_ev_ang = forces_ev_ang + df_full[:nq, :]
                        forces_mm_ev_ang = df_full[nq:, :]

            resp = {
                "status": "ok",
                "energy_ev": float(energy_ev),
            }
            # Keep arrays as numpy when possible; _send_msg_auto will handle
            # serialization (binary keeps ndarray, JSON falls back to .tolist()).
            if forces_ev_ang is not None:
                resp["forces_ev_ang"] = np.asarray(forces_ev_ang, dtype=np.float64)
            else:
                resp["forces_ev_ang"] = None
            if hess_ev_ang2 is not None:
                resp["hessian_ev_ang2"] = np.asarray(hess_ev_ang2, dtype=np.float64)
            else:
                resp["hessian_ev_ang2"] = None
            if forces_mm_ev_ang is not None:
                resp["forces_mm_ev_ang"] = np.asarray(forces_mm_ev_ang, dtype=np.float64)
            return resp
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def server_is_alive(socket_path, timeout=5.0):
    """Return True if a server is responding at *socket_path*."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(os.path.abspath(socket_path))
        _send_msg(sock, {"action": "ping"})
        resp = _recv_msg(sock)
        return resp is not None and resp.get("status") == "ok"
    except Exception:
        return False
    finally:
        sock.close()


def send_shutdown(socket_path, timeout=10.0):
    """Send a shutdown command to the server."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(os.path.abspath(socket_path))
        _send_msg(sock, {"action": "shutdown"})
        return _recv_msg(sock)
    finally:
        sock.close()


def client_evaluate(
    socket_path,
    symbols,
    coords_ang,
    charge,
    multiplicity,
    need_forces,
    need_hessian,
    hessian_mode,
    hessian_step,
    timeout=600.0,
    mm_coords_ang=None,
    mm_charges=None,
):
    """Connect to a running MLIPServer and request an evaluation.

    Returns ``(energy_ev, forces_ev_ang_or_None, hessian_ev_ang2_or_None)``.
    """
    socket_path = os.path.abspath(socket_path)
    if not os.path.exists(socket_path):
        raise ServerError(
            "Server socket not found: {}. Is the server running?".format(socket_path)
        )

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
        meta = {
            "action": "evaluate",
            "symbols": list(symbols),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "need_forces": bool(need_forces),
            "need_hessian": bool(need_hessian),
            "hessian_mode": str(hessian_mode),
            "hessian_step": float(hessian_step),
        }
        arrays = [
            ("coords_ang", np.asarray(coords_ang, dtype=np.float64)),
        ]
        if mm_coords_ang is not None:
            arrays.append(("mm_coords_ang", np.asarray(mm_coords_ang, dtype=np.float64)))
        if mm_charges is not None:
            arrays.append(("mm_charges", np.asarray(mm_charges, dtype=np.float64)))
        _send_msg_bin(sock, meta, arrays)
        response, _ = _recv_msg_auto(sock)
    finally:
        sock.close()

    if response is None:
        raise ServerError("No response from server")
    if response.get("status") != "ok":
        raise ServerError("Server error: {}".format(response.get("message", "unknown")))

    energy_ev = float(response["energy_ev"])
    fv = response.get("forces_ev_ang")
    forces_ev_ang = (
        np.asarray(fv, dtype=np.float64)
        if fv is not None
        else None
    )
    hess_ev_ang2 = (
        np.asarray(response["hessian_ev_ang2"], dtype=np.float64)
        if response.get("hessian_ev_ang2") is not None
        else None
    )
    return energy_ev, forces_ev_ang, hess_ev_ang2


class PersistentClient(object):
    """Reusable client that keeps the socket open across evaluations.

    Usage::

        client = PersistentClient("/tmp/mlip_server.sock")
        for step in range(n_steps):
            energy, forces, hess = client.evaluate(...)
        client.close()

    Also works as a context manager::

        with PersistentClient(socket_path) as client:
            energy, forces, hess = client.evaluate(...)
    """

    def __init__(self, socket_path, timeout=600.0):
        self.socket_path = os.path.abspath(socket_path)
        self.timeout = float(timeout)
        self._sock = None

    def _ensure_connected(self):
        if self._sock is not None:
            return
        if not os.path.exists(self.socket_path):
            raise ServerError(
                "Server socket not found: {}. Is the server running?".format(
                    self.socket_path
                )
            )
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect(self.socket_path)

    def _reset(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

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
        mm_coords_ang=None,
        mm_charges=None,
    ):
        """Send an evaluate request over the persistent connection."""
        self._ensure_connected()
        meta = {
            "action": "evaluate",
            "symbols": list(symbols),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "need_forces": bool(need_forces),
            "need_hessian": bool(need_hessian),
            "hessian_mode": str(hessian_mode),
            "hessian_step": float(hessian_step),
        }
        arrays = [
            ("coords_ang", np.asarray(coords_ang, dtype=np.float64)),
        ]
        if mm_coords_ang is not None:
            arrays.append(("mm_coords_ang", np.asarray(mm_coords_ang, dtype=np.float64)))
        if mm_charges is not None:
            arrays.append(("mm_charges", np.asarray(mm_charges, dtype=np.float64)))
        try:
            _send_msg_bin(self._sock, meta, arrays)
            response, _ = _recv_msg_auto(self._sock)
        except Exception:
            self._reset()
            raise

        if response is None:
            self._reset()
            raise ServerError("No response from server")
        if response.get("status") != "ok":
            raise ServerError(
                "Server error: {}".format(response.get("message", "unknown"))
            )

        energy_ev = float(response["energy_ev"])
        fv = response.get("forces_ev_ang")
        forces_ev_ang = (
            np.asarray(fv, dtype=np.float64)
            if fv is not None
            else None
        )
        hv = response.get("hessian_ev_ang2")
        hess_ev_ang2 = (
            np.asarray(hv, dtype=np.float64)
            if hv is not None
            else None
        )
        return energy_ev, forces_ev_ang, hess_ev_ang2

    def close(self):
        """Close the persistent connection."""
        self._reset()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Auto-start helpers
# ---------------------------------------------------------------------------


def auto_server_socket(args, parent_pid=None):
    """Compute a deterministic socket path from evaluator arguments."""
    key_parts = [str(getattr(args, "model", "default"))]
    if hasattr(args, "device"):
        key_parts.append(str(args.device))
    if hasattr(args, "task"):
        key_parts.append(str(args.task))
    if hasattr(args, "precision"):
        key_parts.append(str(args.precision))
    if parent_pid is None:
        parent_pid = os.getppid()
    key_parts.append("ppid={}".format(int(parent_pid)))

    key = "_".join(key_parts)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    uid = os.getuid()
    return os.path.join(
        tempfile.gettempdir(),
        "mlip_server_{uid}_{hash}.sock".format(uid=uid, hash=h),
    )


def _build_serve_argv(
    executable, custom_args, socket_path, idle_timeout, parent_pid=None
):
    """Build the argv for spawning the server subprocess."""
    cmd = [sys.executable, executable]
    cmd.extend(custom_args)
    cmd.extend(["--serve", "--server-socket", socket_path])
    if idle_timeout is not None:
        cmd.extend(["--server-idle-timeout", str(int(idle_timeout))])
    if parent_pid is not None:
        cmd.extend(["--server-parent-pid", str(int(parent_pid))])
    return cmd


def ensure_server(
    executable,
    custom_args,
    socket_path,
    idle_timeout=600,
    parent_pid=None,
):
    """Ensure a server is running at *socket_path*.

    If no server is alive, spawn one as a background subprocess and wait
    until it becomes ready.

    Returns True if the server is ready, False otherwise.
    """
    if server_is_alive(socket_path, timeout=2.0):
        return True

    # Clean up stale socket file
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    cmd = _build_serve_argv(
        executable=executable,
        custom_args=custom_args,
        socket_path=socket_path,
        idle_timeout=idle_timeout,
        parent_pid=parent_pid,
    )
    print(
        "[mlip-client] Starting server: {}".format(" ".join(cmd)),
        file=sys.stderr,
        flush=True,
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            start_new_session=True,
        )
    except Exception as exc:
        print(
            "[mlip-client] WARNING: Failed to start server: {}".format(exc),
            file=sys.stderr,
            flush=True,
        )
        return False

    startup_timeout = 300
    waited = 0
    while waited < startup_timeout:
        if proc.poll() is not None:
            print(
                "[mlip-client] WARNING: Server process exited unexpectedly (code={}).".format(
                    proc.returncode
                ),
                file=sys.stderr,
                flush=True,
            )
            return False

        if server_is_alive(socket_path, timeout=1.0):
            print("[mlip-client] Server is ready.", file=sys.stderr, flush=True)
            return True

        time.sleep(1)
        waited += 1

    print(
        "[mlip-client] WARNING: Server did not become ready within {}s.".format(
            startup_timeout
        ),
        file=sys.stderr,
        flush=True,
    )
    return False
