from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_MODBUS_TIMEOUT_SEC = 0.5


def _normalize_force_vector(force: Sequence[float]) -> np.ndarray:
    data = np.asarray(force, dtype=float).reshape(-1)
    if data.shape != (6,):
        raise ValueError(f"InspireTac force expects 6 values, got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("InspireTac force values must be finite.")
    if np.any(data < 0):
        raise ValueError("InspireTac force values must be non-negative.")
    return data.astype(int)


def _response_is_error(response: object) -> bool:
    if response is None:
        return True
    is_error = getattr(response, "isError", None)
    return bool(is_error()) if callable(is_error) else False


def open_modbus(ip: str, port: int, *, timeout: float = DEFAULT_MODBUS_TIMEOUT_SEC):
    from pymodbus.client.sync import ModbusTcpClient

    client = ModbusTcpClient(ip, int(port), timeout=float(timeout))
    if not client.connect():
        client.close()
        raise RuntimeError(f"failed to connect InspireTac Modbus {ip}:{port}")
    return client


def write_register(client: object, address: int, values: Sequence[int]) -> None:
    payload = list(values)
    response = client.write_registers(int(address), payload)
    if _response_is_error(response):
        raise RuntimeError(
            f"Modbus write_registers failed address={address} values={payload}: {response}"
        )


def read_register(client: object, address: int, count: int) -> list[int]:
    response = client.read_holding_registers(int(address), int(count))
    if _response_is_error(response):
        raise RuntimeError(
            f"Modbus read_holding_registers failed address={address} count={count}: {response}"
        )
    return list(response.registers)


def write6(client: object, reg: int, val: Sequence[int]) -> None:
    write_register(client, int(reg), [int(item) & 0xFFFF for item in val[:6]])


def read6(client: object, reg: int, *, n_short: int = 6, use_byte: bool = False) -> list[int]:
    values = read_register(client, int(reg), int(n_short))
    if len(values) < int(n_short):
        raise RuntimeError("Modbus read returned insufficient data.")
    if not use_byte:
        return values
    results: list[int] = []
    for value in values:
        results.append(int(value) & 0xFF)
        results.append((int(value) >> 8) & 0xFF)
    return results


class InspireTacModbus:
    def __init__(
        self,
        ip: str = "192.168.12.210",
        port: int = 6000,
        max_fingertip_force: int = 800,
        timeout: float = DEFAULT_MODBUS_TIMEOUT_SEC,
    ):
        self.ip = str(ip)
        self.port = int(port)
        self.timeout = float(timeout)
        self.max_fingertip_force = int(max_fingertip_force)
        if self.max_fingertip_force < 0:
            raise ValueError("max_fingertip_force must be non-negative.")
        self.client = open_modbus(self.ip, self.port, timeout=self.timeout)
        self.last_force_command: np.ndarray | None = None
        self.regdict = {
            "ID": 1000,
            "baudrate": 1001,
            "clearErr": 1004,
            "forceClb": 1009,
            "defaultSpeedSet": 1032,
            "defaultForceSet": 1044,
            "angleSet": 1486,
            "forceSet": 1498,
            "speedSet": 1522,
            "angleAct": 1546,
            "forceAct": 1582,
            "errCode": 1606,
            "statusCode": 1612,
            "temp": 1618,
            "actionSeq": 2320,
            "actionRun": 2322,
        }

        self.tac_chunks_start = [(3000 + index * 114) for index in range(18)] + [5052]
        self.tac_chunks_count = [57] * 18 + [36]
        self.tac_region_names = [
            "little_1",
            "little_2",
            "little_3",
            "ring_1",
            "ring_2",
            "ring_3",
            "middle_1",
            "middle_2",
            "middle_3",
            "index_1",
            "index_2",
            "index_3",
            "thumb_1",
            "thumb_2",
            "thumb_3",
            "thumb_4",
            "palm",
        ]
        self.tac_region_sizes = [
            (3, 3),
            (12, 8),
            (10, 8),
            (3, 3),
            (12, 8),
            (10, 8),
            (3, 3),
            (12, 8),
            (10, 8),
            (3, 3),
            (12, 8),
            (10, 8),
            (3, 3),
            (12, 8),
            (3, 3),
            (12, 8),
            (8, 14),
        ]
        self.tac_region_start_idxs = [0]
        for size in self.tac_region_sizes:
            self.tac_region_start_idxs.append(
                self.tac_region_start_idxs[-1] + int(size[0]) * int(size[1])
            )
        self.chunk_descriptors: list[dict[str, int]] = []
        cursor = 0
        for chunk_idx, (start_addr, count) in enumerate(
            zip(self.tac_chunks_start, self.tac_chunks_count)
        ):
            self.chunk_descriptors.append(
                {
                    "chunk_idx": int(chunk_idx),
                    "reg_start": int(start_addr),
                    "count": int(count),
                    "buf_start": int(cursor),
                    "buf_end": int(cursor + count),
                }
            )
            cursor += int(count)

        self.region_to_chunks: dict[str, list[int]] = {}
        for index, name in enumerate(self.tac_region_names):
            region_start = self.tac_region_start_idxs[index]
            region_end = self.tac_region_start_idxs[index + 1]
            chunk_indices: list[int] = []
            for chunk in self.chunk_descriptors:
                if region_start < chunk["buf_end"] and chunk["buf_start"] < region_end:
                    chunk_indices.append(chunk["chunk_idx"])
            self.region_to_chunks[name] = chunk_indices

        self.get_default_speed_force()
        self.set_force([self.max_fingertip_force] * 6)
        print("Startup force set:", self.last_force_command.tolist())

    def get_default_speed_force(self) -> tuple[list[int], list[int]]:
        speed = read6(self.client, self.regdict["defaultSpeedSet"], n_short=6, use_byte=False)
        force = read6(self.client, self.regdict["defaultForceSet"], n_short=6, use_byte=False)
        print("Default speed set:", speed)
        print("Default force set:", force)
        return speed, force

    def calib_joint_force(self) -> None:
        write_register(self.client, self.regdict["forceClb"], [1])

    def read_tac_data(
        self,
        target_regions: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if target_regions is None:
            chunks_to_read = range(len(self.chunk_descriptors))
            regions_to_process = self.tac_region_names
        else:
            chunk_set: set[int] = set()
            for name in target_regions:
                if name in self.region_to_chunks:
                    chunk_set.update(self.region_to_chunks[name])
                else:
                    print(f"Warning: tactile region {name} not found")
            chunks_to_read = sorted(chunk_set)
            regions_to_process = [name for name in target_regions if name in self.tac_region_names]

        full_results = np.zeros(1062, dtype=float)
        for chunk_index in chunks_to_read:
            chunk = self.chunk_descriptors[int(chunk_index)]
            values = read_register(self.client, chunk["reg_start"], chunk["count"])
            if len(values) == chunk["count"]:
                full_results[chunk["buf_start"] : chunk["buf_end"]] = values

        tac_vec = full_results / 4096.0
        tac_dict: dict[str, np.ndarray] = {}
        for name in regions_to_process:
            index = self.tac_region_names.index(name)
            region_size = self.tac_region_sizes[index]
            start = self.tac_region_start_idxs[index]
            end = self.tac_region_start_idxs[index + 1]
            region_data = tac_vec[start:end].reshape(region_size)
            if name.endswith("_1") or name == "thumb_3" or name == "palm":
                region_data = region_data.reshape(region_size[1], region_size[0]).T
            elif name == "thumb_4":
                region_data = region_data[::-1, ::-1]
            tac_dict[name] = region_data
        return tac_vec, tac_dict

    def read_joint_pos(self) -> np.ndarray:
        data = read6(self.client, self.regdict["angleAct"], n_short=6, use_byte=False)
        return np.asarray(data, dtype=float) / 1000.0

    def set_joint_pos(self, joint_pos: Sequence[float]) -> None:
        data = (np.asarray(joint_pos, dtype=float) * 1000.0).astype(int)
        write6(self.client, self.regdict["angleSet"], data.tolist())

    def set_force(self, force: Sequence[float]) -> None:
        data = _normalize_force_vector(force)
        write6(self.client, self.regdict["forceSet"], data.tolist())
        self.last_force_command = data.copy()

    def close(self) -> None:
        self.client.close()
