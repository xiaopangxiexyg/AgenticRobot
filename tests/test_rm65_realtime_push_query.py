from __future__ import annotations

import pytest

from scripts.probes.query_rm65_realtime_push import query_configs


@pytest.mark.parametrize("failure", [None, "query", "close"])
def test_query_only_reads_once_and_closes_each_owned_handle(failure):
    calls = []

    class Arm:
        nSocket = 1

        def __init__(self, ip):
            self.ip = ip

        def Get_Realtime_Push_Config(self):
            calls.append((self.ip, "query"))
            if failure == "query":
                raise RuntimeError("query failed")
            return 0, {"enable": True, "port": 8089}

        def Arm_Socket_Close(self):
            calls.append((self.ip, "close"))
            return 1 if failure == "close" else 0

        def __getattr__(self, name):
            raise AssertionError(f"unexpected SDK action: {name}")

    ips = {"left": "left-controller", "right": "right-controller"}
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            query_configs(ips, Arm)
        assert calls == [("left-controller", "query"), ("left-controller", "close")]
    else:
        result = query_configs(ips, Arm)
        assert set(result) == {"left", "right"}
        assert calls == [
            (side, operation)
            for side in ips.values()
            for operation in ("query", "close")
        ]


@pytest.mark.parametrize("occupied", [False, True])
def test_passive_receive_does_not_reuse_an_occupied_port_or_connect_sdk(occupied):
    from scripts.probes.query_rm65_realtime_push import listen_existing

    clock = [0.0]
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def bind(self, address):
            calls.append(address)
            if occupied:
                raise OSError("port occupied")

        def settimeout(self, duration):
            assert 0 < duration <= 0.5

        def recvfrom(self, size):
            clock[0] += 0.5
            return b'{"joint_status":{"joint_position":[1,2,3,4,5,6]}}', (
                "right-ip",
                8090,
            )

    if occupied:
        with pytest.raises(OSError, match="occupied"):
            listen_existing(
                {"left": "left-ip", "right": "right-ip"},
                8090,
                socket_factory=lambda *args: Socket(),
                clock=lambda: clock[0],
            )
        assert calls == [("0.0.0.0", 8090), "closed"]
    else:
        report = listen_existing(
            {"left": "left-ip", "right": "right-ip"},
            8090,
            socket_factory=lambda *args: Socket(),
            clock=lambda: clock[0],
        )
        assert report["packet_count_by_side"] == {"left": 0, "right": 20}
        assert report["sdk_connected"] is False
        assert calls[-1] == "closed"
