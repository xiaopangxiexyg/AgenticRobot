# Foot Pad Input

This module maps USB foot pads to a virtual keyboard.  Up to ten pads are
supported: they are calibrated in order and emit `F13` through `F22`
respectively.  Four pads are the default configuration, and two pads are also
supported for compatibility;
the real-teleop GUI maps four pads to left/right Fix-arm and left/right exact
Init-qpos collision-planned motion.  With two pads, only left/right Fix-arm is
emitted by the pedals.  Five pads use the same first-two side-switch actions;
pad 3 advances progress, pad 4 reports the current step as failed, and pad 5 undoes
the last progress advance when a progress profile is ready.  Before profile
selection, only pads 1/2 are active and pads 3/4/5 remain fail-closed.  Foot
pads are optional, additive inputs: their topology or
connection state never disables the configured GUI/keyboard Fix, Init, or
exact-qpos controls. Other counts can still be calibrated by this low-level
service, but the real-teleop foot-pad adapter falls back to manual pedal mapping
for those layouts.
The five-pad topology follows the two-pad side-switch contract for its first two
pedals: pad 1 enters left-arm-only teleop (Fix right) and pad 2 enters
right-arm-only teleop (Fix left).  A starts or stops collection, while B/C use
the same dedicated two-pad dispatch and Hold/Home transitions.  Init remains
available through G/GUI rather than being silently assigned to pads 3/4/5.
The low-level service accepts any count from 1 through 10.  The source devices
are identified by their udev `ID_PATH`, not `/dev/input/eventN`.

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

For the five-pad layout (Fix prefix plus optional progress actions):

```bash
python -m hardware.foot_pad --count 5
```

For a generic ten-pad layout:

```bash
python -m hardware.foot_pad --count 10
```

Later starts load `teleop_runtime/configs/hardware/foot_pad/latest.json`
automatically.  The process exclusively grabs only the calibrated pads, so their
original key events do not also reach the desktop.  It waits and reconnects when
a configured pad is removed and reattached.  The virtual keyboard advertises
exactly the configured topology: `F13`/`F14` for a two-pad config, `F13`–`F16`
for a four-pad config, or the first N keys through `F22` for other counts.  The
current real-teleop adapter retains its action semantics for the two-, four-,
and five-pad layouts.  Five-pad progress actions require an active progress
configuration; other counts remain available through this low-level service
and are not assigned new arm actions automatically.
Restart the service after changing `--count` or recalibrating.

The RealTeleop five-pedal safety test is also persistent.  After all five
pedals have been observed once, RealTeleop writes
`teleop_runtime/configs/hardware/foot_pad/five_pedal_test_receipt.json`.  The
receipt is tied to the exact physical-to-virtual assignment and to the current
five-pedal semantic mapping, and is shared by position and impedance control.
Later runs restore the verified state automatically.  Recalibrating, reordering,
or replacing a pedal invalidates the fingerprint and requires one new complete
test.  The forwarding service must still be running; the receipt replaces only
the repeated operator test, not device discovery or connection.

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
