#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主要改动：
1. 两条 TCP 连接分离
   - cmd_sock: 只发 {"q":[...]}，不等回复
   - rpc_sock: 只做 get/ping/stop 等 request-reply
2. 支持后台状态轮询线程
   - start_state_poller()
   - get_state_cached()
3. 建议采集主循环里不要再同步调用 get_state()，
   而是使用 get_state_cached()

示例：
python fr3_joint_control_client.py --host 192.168.20.172 --port 5558 --timeout_s 2.0
"""

import argparse
import json
import socket
import threading
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


def recv_line(sock: socket.socket, buf: bytearray, timeout_s: float) -> Tuple[Optional[str], bytearray]:
    old_timeout = sock.gettimeout()
    sock.settimeout(timeout_s)
    try:
        while True:
            nl = buf.find(b"\n")
            if nl != -1:
                line = bytes(buf[:nl]).decode("utf-8", errors="replace").strip()
                del buf[: nl + 1]
                if not line:
                    continue
                return line, buf

            try:
                chunk = sock.recv(4096)
            except socket.timeout as e:
                raise TimeoutError(
                    f"recv_line timeout after {timeout_s:.2f}s, buffered={len(buf)} bytes"
                ) from e

            if not chunk:
                return None, buf
            buf.extend(chunk)
    finally:
        sock.settimeout(old_timeout)


def send_json(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


class FR3JointClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 5.0,
        retry_times: int = 1,
        retry_wait_s: float = 0.05,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.retry_times = int(retry_times)
        self.retry_wait_s = float(retry_wait_s)

        # 两条连接：
        # 1) cmd_sock：只发 q，不等待回复
        # 2) rpc_sock：只做 get/ping/stop 等 request-reply
        self.cmd_sock: Optional[socket.socket] = None
        self.rpc_sock: Optional[socket.socket] = None
        self.rpc_buf = bytearray()

        self._cmd_lock = threading.RLock()
        self._rpc_lock = threading.RLock()

        # 后台状态轮询
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_hz: float = 20.0
        self._warn_stale_after_s: float = 0.2

        # 状态缓存
        self._state_lock = threading.Lock()
        self._cached_state: Optional[Dict[str, Any]] = None
        self._cached_state_recv_mono: float = 0.0
        self._last_poll_error_t: float = 0.0

    def _create_socket(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        return sock

    def connect(self):
        with self._cmd_lock:
            if self.cmd_sock is None:
                self.cmd_sock = self._create_socket()

        with self._rpc_lock:
            if self.rpc_sock is None:
                self.rpc_sock = self._create_socket()
                self.rpc_buf = bytearray()

        return self

    def _close_cmd_locked(self):
        if self.cmd_sock is not None:
            try:
                self.cmd_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.cmd_sock.close()
            except Exception:
                pass
        self.cmd_sock = None

    def _close_rpc_locked(self):
        if self.rpc_sock is not None:
            try:
                self.rpc_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.rpc_sock.close()
            except Exception:
                pass
        self.rpc_sock = None
        self.rpc_buf = bytearray()

    def _reconnect_cmd_locked(self):
        self._close_cmd_locked()
        self.cmd_sock = self._create_socket()

    def _reconnect_rpc_locked(self):
        self._close_rpc_locked()
        self.rpc_sock = self._create_socket()
        self.rpc_buf = bytearray()

    def close(self):
        self.stop_state_poller()
        with self._cmd_lock:
            self._close_cmd_locked()
        with self._rpc_lock:
            self._close_rpc_locked()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------------------
    # command socket: q only
    # ---------------------------

    def send_q(self, q: List[float]) -> None:
        req = {"q": q}
        last_err = None

        for attempt in range(self.retry_times + 1):
            with self._cmd_lock:
                try:
                    if self.cmd_sock is None:
                        self._reconnect_cmd_locked()
                    assert self.cmd_sock is not None
                    send_json(self.cmd_sock, req)
                    return
                except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError) as e:
                    last_err = e
                    try:
                        self._reconnect_cmd_locked()
                    except Exception as reconnect_err:
                        last_err = reconnect_err

            if attempt < self.retry_times:
                time.sleep(self.retry_wait_s)

        raise RuntimeError(f"send_q failed after retries, err={last_err}")

    # ---------------------------
    # rpc socket: get / ping / stop / quit
    # ---------------------------

    def _request_json_rpc(self, req: dict, expect_reply: bool = True) -> Optional[dict]:
        last_err = None

        for attempt in range(self.retry_times + 1):
            with self._rpc_lock:
                try:
                    if self.rpc_sock is None:
                        self._reconnect_rpc_locked()

                    assert self.rpc_sock is not None
                    send_json(self.rpc_sock, req)

                    if not expect_reply:
                        return None

                    line, self.rpc_buf = recv_line(self.rpc_sock, self.rpc_buf, timeout_s=self.timeout_s)
                    if not line:
                        raise RuntimeError("peer closed connection")

                    resp = json.loads(line)
                    return resp

                except (TimeoutError, socket.timeout, ConnectionResetError,
                        BrokenPipeError, OSError, json.JSONDecodeError) as e:
                    last_err = e
                    try:
                        self._reconnect_rpc_locked()
                    except Exception as reconnect_err:
                        last_err = reconnect_err

            if attempt < self.retry_times:
                time.sleep(self.retry_wait_s)

        raise RuntimeError(f"rpc request failed after retries, req={req}, err={last_err}")

    def get_state(self) -> dict:
        resp = self._request_json_rpc({"cmd": "get"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad get resp: {resp}")
        return resp

    def stop(self) -> None:
        resp = self._request_json_rpc({"cmd": "stop"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad stop resp: {resp}")

    def ping(self) -> dict:
        resp = self._request_json_rpc({"cmd": "ping"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad ping resp: {resp}")
        return resp

    def quit_server(self) -> None:
        resp = self._request_json_rpc({"cmd": "quit"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad quit resp: {resp}")

    # ---------------------------
    # background state poller
    # ---------------------------

    def start_state_poller(self, poll_hz: float = 20.0, warn_stale_after_s: float = 0.2):
        self.connect()

        self._poll_hz = float(poll_hz)
        self._warn_stale_after_s = float(warn_stale_after_s)

        if self._poll_thread is not None and self._poll_thread.is_alive():
            return

        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._state_poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_state_poller(self):
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._poll_thread = None

    def _state_poll_loop(self):
        dt = 1.0 / max(self._poll_hz, 1e-6)
        next_t = time.monotonic()

        while not self._poll_stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += dt

            try:
                st = self.get_state()
                recv_t = time.monotonic()
                with self._state_lock:
                    self._cached_state = st
                    self._cached_state_recv_mono = recv_t
            except Exception as e:
                now2 = time.monotonic()
                if (now2 - self._last_poll_error_t) > 1.0:
                    print(f"[FR3StatePoller] warn: get_state failed: {e}")
                    self._last_poll_error_t = now2

                # 避免下一轮瞬间疯狂追赶
                next_t = time.monotonic() + dt

    def wait_until_state_ready(self, timeout_s: float = 2.0) -> bool:
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_s:
            with self._state_lock:
                ready = self._cached_state is not None
            if ready:
                return True
            time.sleep(0.01)
        return False

    def get_state_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[dict, float]:
        """
        返回 (state_dict, age_s)

        age_s 是“客户端视角下缓存距现在多久”。
        若 server 也返回 age_s，则这里只是 client 缓存年龄；
        server 侧年龄可从 state_dict["age_s"] 读取。
        """
        with self._state_lock:
            st = None if self._cached_state is None else dict(self._cached_state)
            recv_t = float(self._cached_state_recv_mono)

        if st is not None:
            age_s = max(0.0, time.monotonic() - recv_t)
            if (max_age_s is None) or (age_s <= max_age_s):
                return st, age_s

        if fallback_to_sync:
            st = self.get_state()
            age_s = 0.0
            with self._state_lock:
                self._cached_state = st
                self._cached_state_recv_mono = time.monotonic()
            return st, age_s

        raise RuntimeError("cached FR3 state unavailable or too stale")

    def get_joint_state_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[np.ndarray, float]:
        st, age = self.get_state_cached(max_age_s=max_age_s, fallback_to_sync=fallback_to_sync)
        q = st.get("q", None)
        if q is None:
            raise RuntimeError(f"cached state has no q: {st}")
        return np.asarray(q, dtype=np.float32), age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="192.168.50.2")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--timeout_s", type=float, default=2.0)
    args = ap.parse_args()

    with FR3JointClient(args.host, args.port, timeout_s=args.timeout_s) as c:
        print("[ping ]", c.ping())

        st0 = c.get_state()
        print("[sync get] age(server) =", st0.get("age_s", None), "q =", st0.get("q", None))

        c.start_state_poller(poll_hz=20.0, warn_stale_after_s=0.2)
        ok = c.wait_until_state_ready(timeout_s=2.0)
        print("[poll ] ready =", ok)

        st_cached, age_cached = c.get_state_cached(max_age_s=1.0, fallback_to_sync=True)
        print(
            "[cached ] client_cache_age =", age_cached,
            "server_age =", st_cached.get("age_s", None),
            "q =", st_cached.get("q", None)
        )

        q0 = np.array(st_cached["q"], dtype=np.float64)
        q_des = q0.copy()
        q_des[3] = q_des[3] - 0.1

        print("[send_q] sending one target...")
        c.send_q(q_des.tolist())

        time.sleep(0.5)

        st1, age1 = c.get_state_cached(max_age_s=1.0, fallback_to_sync=True)
        print(
            "[after ] client_cache_age =", age1,
            "server_age =", st1.get("age_s", None),
            "q =", st1.get("q", None)
        )

if __name__ == "__main__":
    main()