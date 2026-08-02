# UMI Control Station boundary

This package owns UMI presentation policy, operator-intent mapping, view
models, and application workflow coordination. It must not read VIVE/Q6/angle
trigger/camera devices, dispatch gripper commands, own calibration validity,
or start/finalize an episode. Those operations remain behind typed UMI and
collection runtime ports.

The current controller import is an explicit migration seam used only to
consume pure control-availability policy. Boundary tests prevent that reverse
dependency from spreading while the UMI coordinator is extracted.

UMI uses the same `control_station.web.calibration_web_ui` transport as
RealTeleop. UMI-specific control availability and event mapping stay in this
feature package, rather than forking the HTTP server or browser state schema.
