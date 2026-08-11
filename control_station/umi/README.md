# UMI Control Station boundary

This package owns UMI presentation policy, operator-intent mapping, and view
models. It must not read VIVE/Q6/angle-trigger/camera devices, dispatch gripper
commands, own calibration validity, or start/finalize an episode. Those
operations remain behind typed UMI and collection-runtime ports.

The sibling `control_station/umi_application.py` is the concrete application
leaf that binds those ports to the Web UI. Lower UMI modules depend only on the
operator-policy protocol declared in `teleop_runtime`; they do not import this
package. The historic `teleop_runtime.collection.umi.app` path is a thin
compatibility import for the existing command and callers.

UMI uses the same `control_station.web.calibration_web_ui` transport as
RealTeleop. UMI-specific control availability and event mapping stay in this
feature package, rather than forking the HTTP server or browser state schema.
