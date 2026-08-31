# RM65 API1 vendor snapshot retirement — 2026-08-31

## Decision

Remove `hardware/rm65/robotic_arm_package/` from the versioned source tree.
The directory was the old RealMan API1 Python wrapper and binary snapshot. It
is not a compatibility surface used by the current runtime.

The supported replacement is:

- project compatibility API: `hardware.rm65.rm_api2_adapter.Arm`;
- bundled vendor API: `hardware.rm65.Robotic_Arm` (RealMan API2 1.1.5).

No retired source was moved into the root `legacy/` directory. That directory
remains strictly local and untracked.

## Evidence

At `development` commit `8ed16b945691cb6dfd51adbed6579b38af81c1df`:

- no production Python module, test, or current entrypoint imported
  `hardware.rm65.robotic_arm_package`;
- the only current references were three environment/dependency descriptions,
  updated with this retirement;
- the old wrapper hard-coded `libRM_Base.so`, while the versioned directory
  contained only `libRM_Base.so.1.0.0`, so its documented import failed before
  any device connection;
- `hardware.rm65.rm_api2_adapter` imported successfully in the supported
  `dexteleop310` environment without connecting to a device;
- current RM65 controllers, backends, calibration utilities, impedance
  control, and official IK use the API2 adapter or API2 vendor package.

The dated annotation source projection under
`release/refinement_annotation_20260808_c959d1d38695/` is immutable release
provenance rather than a current hardware dependency. Its copied historical
environment text was not rewritten because doing so would invalidate the
projection receipt; the bundle contains no RM65 vendor code and is not an
executable hardware caller.

## Removed versioned paths

- `hardware/rm65/robotic_arm_package/robotic_arm.py`
- `hardware/rm65/robotic_arm_package/log_setting.py`
- `hardware/rm65/robotic_arm_package/libRM_Base.so.1.0.0`

The removed content remains recoverable from Git history at the development
commit named above. Restoring it as an active dependency requires a new vendor
provenance review, an explicit API ownership decision, and current import and
hardware compatibility validation.

## Safety boundary

This retirement changes no API2 adapter, backend, motion, safety, IK, force,
dispatch, calibration, or device configuration logic. Validation is import-
and contract-only; no simulator, RM65 controller, or other device is started.
