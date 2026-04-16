#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Iterable, List


def _parse_int_list(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for value in values:
        if "," in value:
            parts = [x.strip() for x in value.split(",") if x.strip()]
        else:
            parts = [value]
        for part in parts:
            out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a serial port with Dynamixel Protocol 2.0 ping to identify "
            "whether the port/baudrate/id settings match the exoskeleton bus."
        )
    )
    parser.add_argument("--port", type=str, required=True, help="Serial port to probe, e.g. /dev/ttyUSB0")
    parser.add_argument(
        "--baudrates",
        nargs="+",
        default=["57600", "115200", "1000000"],
        help="Baudrates to try. Space or comma separated.",
    )
    parser.add_argument("--id-start", type=int, default=0, help="Start ID for scan")
    parser.add_argument("--id-end", type=int, default=13, help="End ID for scan")
    parser.add_argument("--protocol", type=float, default=2.0, help="Dynamixel protocol version")
    parser.add_argument("--timeout-ms", type=int, default=100, help="Per-packet timeout in ms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import dynamixel_sdk as dxl

    baudrates = _parse_int_list(args.baudrates)
    if args.id_end < args.id_start:
        raise ValueError("--id-end must be >= --id-start")

    port_handler = dxl.PortHandler(args.port)
    packet_handler = dxl.PacketHandler(float(args.protocol))

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open port: {args.port}")

    try:
        try:
            port_handler.setPacketTimeoutMillis(int(args.timeout_ms))
        except Exception:
            pass

        print(f"Probe port: {args.port}")
        print(f"Protocol : {args.protocol}")
        print(f"ID range : {args.id_start}..{args.id_end}")
        print(f"Baudrates: {baudrates}")
        print("")

        any_success = False
        for baud in baudrates:
            print(f"=== baudrate={baud} ===")
            if not port_handler.setBaudRate(int(baud)):
                print("setBaudRate failed")
                print("")
                continue

            success_ids: List[int] = []
            bad_status_count = 0
            no_status_count = 0
            other_errors = {}

            for dxl_id in range(args.id_start, args.id_end + 1):
                try:
                    model, comm, err = packet_handler.ping(port_handler, dxl_id)
                except Exception as exc:
                    print(f"id={dxl_id:02d} exception: {type(exc).__name__}: {exc}")
                    continue

                if comm == dxl.COMM_SUCCESS and err == 0:
                    any_success = True
                    success_ids.append(dxl_id)
                    print(f"id={dxl_id:02d} OK model={model}")
                    continue

                comm_msg = packet_handler.getTxRxResult(comm)
                err_msg = packet_handler.getRxPacketError(err) if err else ""

                if "Incorrect status packet" in comm_msg:
                    bad_status_count += 1
                elif "There is no status packet" in comm_msg:
                    no_status_count += 1
                else:
                    other_errors[comm_msg] = other_errors.get(comm_msg, 0) + 1

                if err_msg:
                    print(f"id={dxl_id:02d} comm={comm_msg} err={err_msg}")

            if success_ids:
                print(f"online ids: {success_ids}")
            else:
                print("online ids: []")

            print(
                "summary: "
                f"incorrect_status={bad_status_count}, "
                f"no_status={no_status_count}, "
                f"other={other_errors}"
            )
            print("")

        if not any_success:
            print("No Dynamixel motors were detected.")
            print("Interpretation:")
            print("- If you mostly see 'Incorrect status packet', the port is likely not a Dynamixel bus at all.")
            print("- If you mostly see 'There is no status packet', the port may be right but baudrate/IDs/wiring are wrong.")
            print("- This repo expects the exoskeleton arm bus to be Protocol 2.0, usually 57600 baud, IDs 0..13.")
    finally:
        port_handler.closePort()


if __name__ == "__main__":
    main()
