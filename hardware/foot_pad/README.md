# Foot Pad Input

This module maps one to four USB foot pads to a virtual keyboard.  Foot pads
are calibrated in order and emit `F13`, `F14`, `F15`, and `F16` respectively.
The source devices are identified by their udev `ID_PATH`, not `/dev/input/eventN`.

Install the runtime dependency in the environment that runs the module:

```bash
python -m pip install -r hardware/foot_pad/requirements.txt
```

Start with an explicit count.  If no valid mapping exists, this enters terminal
calibration and asks for foot pads 1 through N in sequence:

```bash
python -m hardware.foot_pad --count 2
```

Later starts load `teleop_runtime/configs/hardware/foot_pad/latest.json`
automatically.  The process exclusively grabs only the calibrated pads, so their
original key events do not also reach the desktop.  It waits and reconnects when
a configured pad is removed and reattached.

Useful commands:

```bash
python -m hardware.foot_pad --check
python -m hardware.foot_pad --list-devices
python -m hardware.foot_pad --count 2 --recalibrate
```

The account must be able to read `/dev/input/event*`, and create `/dev/uinput`.
On Linux this usually means an input/uinput udev permission rule or running the
process through the project's approved hardware runtime environment.
