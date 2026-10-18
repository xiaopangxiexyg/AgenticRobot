"""Operator-confirmed B601-RS gravity compensation defaults.

This module is the single source of truth for the numerical compensation
profile shared by the guarded hardware owner and its low-gain probe.  The
gravity-debug switch itself remains an execution-mode choice owned by the
hardware launcher; keeping it out of this numerical profile prevents
simulation or offline tools from accidentally changing their safety semantics.
"""

B601_DEFAULT_KP = 0.0
B601_DEFAULT_KD = 0.0
B601_DEFAULT_DRIVER_KD = (0.0,) * 6
B601_DEFAULT_GRAVITY_SCALE = 1.0
B601_DEFAULT_GRAVITY_JOINT_SCALES = (1.0, 0.96, 1.0, 1.0, 1.0, 1.0)
B601_DEFAULT_GRAVITY_JOINT_OFFSETS_RAD = (0.0,) * 6
B601_DEFAULT_GRAVITY_JOINT_BIAS_NM = (0.0, -0.22, -0.11, -0.08, 0.0, 0.0)
B601_DEFAULT_GRAVITY_COULOMB_FRICTION_NM = (0.0,) * 6
B601_DEFAULT_GRAVITY_COULOMB_VELOCITY_THRESHOLD_RAD_S = 0.02
