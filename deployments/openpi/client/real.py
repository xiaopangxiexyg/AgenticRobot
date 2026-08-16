"""Compatibility entrypoint for the real OpenPI deployment application.

OpenPI protocol, transport, and artifact integrations remain owned by
``deployments.openpi``.  The application that binds them to the real robot
runtime is owned by ``teleop_runtime.deploy``.
"""

from teleop_runtime.deploy.openpi_application import (
    _artifact_path_from_checkpoint,
    _image_key_map_json,
    _normalization_contract_json,
    _rm65_bimanual_dmtac_args,
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
