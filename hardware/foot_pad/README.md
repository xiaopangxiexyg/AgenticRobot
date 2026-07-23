# Foot Pad Input

This module maps USB foot pads to a virtual keyboard.  Four pads are the
default configuration: they are calibrated in order and emit `F13`, `F14`,
`F15`, and `F16` respectively.  Two pads are also supported for compatibility;
the real-teleop GUI maps four pads to left/right Fix-arm and left/right exact
Init-qpos collision-planned motion.  With two pads, only left/right Fix-arm is
emitted by the pedals.  Foot pads are optional, additive inputs: their topology
or connection state never disables the configured GUI/keyboard Fix, Init, or
exact-qpos controls. Other counts can still be calibrated by this low-level
service, but the real-teleop foot-pad adapter falls back to manual pedal mapping
for those layouts.
The source devices are identified by their udev `ID_PATH`, not `/dev/input/eventN`.

Install the runtime dependency in the environment that runs the module:

```bash
python -m pip install -r hardware/foot_pad/requirements.txt
```

Start with four pads (the recommended/default layout).  If no valid mapping
exists, this enters terminal calibration and asks for foot pads 1 through N in
sequence:

```bash
python -m hardware.foot_pad --count 4
```

For the two-pad compatibility layout:

```bash
python -m hardware.foot_pad --count 2
```

Later starts load `teleop_runtime/configs/hardware/foot_pad/latest.json`
automatically.  The process exclusively grabs only the calibrated pads, so their
original key events do not also reach the desktop.  It waits and reconnects when
a configured pad is removed and reattached.  The virtual keyboard advertises
only `F13`/`F14` for a two-pad config (and all four keys for a four-pad config),
so the GUI can identify the live topology and show the correct pedal mapping.
Restart the service after changing `--count` or recalibrating.

Useful commands:

```bash
python -m hardware.foot_pad --check
python -m hardware.foot_pad --list-devices
python -m hardware.foot_pad --count 2 --recalibrate
```

The account must be able to read the physical `/dev/input/event*` nodes, create
`/dev/uinput`, and read the virtual `DexTeleop foot pads` event node. The
virtual node is normally created as `root:input 0660`; without a persistent
rule the forwarding service can report success while the GUI cannot open it.
Install the packaged device-specific udev rule and repair current nodes once:

```bash
sudo -E "$CONDA_PREFIX/bin/python" -m teleop_runtime.configure_devices --fix-permissions
```

`--fix-permissions` installs
`/etc/udev/rules.d/99-dexteleop-foot-pad.rules`. It only makes the named
DexTeleop virtual keyboard world-readable; it does not relax every input
device. Startup probes once. If the virtual keyboard is absent or later
disconnects, the running GUI does not scan automatically; press
“重连失败设备” to request one explicit scan after the node becomes readable.
The retry control is available before runtime Home or in regular Hold, and is
disabled during live arm control, exact-qpos motion, and episode transitions.
One accepted click is latched until that retry finishes, so repeated browser
clicks cannot queue repeated scans.
Discovery filters non-foot-pad event nodes through sysfs before opening evdev.
