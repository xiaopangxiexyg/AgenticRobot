"""OpenPI-specific normalization declarations.

OpenPI's current DexTeleop configs load ``norm_stats.json`` into the server
pipeline.  The robot client sends raw compact state and receives raw compact
actions; it must not apply the q01/q99 transform a second time.  This helper
only constructs the explicit shared contract and never executes the transform.
"""

from __future__ import annotations

import re
from typing import Any

from dataset_contracts.normalization import (
    NormalizationContract,
    NormalizationContractError,
    NormalizationDomainSpec,
)


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def make_openpi_quantile_normalization_contract(
    *,
    stats_ref: str,
    stats_hash: str,
    state_input_space: str = "runtime_state_16_raw",
    state_output_space: str = "openpi_state_q01_q99_m11",
    action_input_space: str = "openpi_action_q01_q99_m11",
    action_output_space: str = "runtime_action_16_raw",
    image_space: str = "runtime_rgb_uint8_hwc",
) -> dict[str, Any]:
    """Build the server-owned q01/q99 contract used by current DexTeleop OpenPI.

    ``stats_ref`` is an artifact-relative reference to the exact
    ``norm_stats.json`` bytes.  The ``#state`` and ``#actions`` selectors are
    part of the reference because OpenPI stores both vectors in that one JSON
    artifact.  ``stats_hash`` is the SHA-256 of the complete file, including
    formatting, as returned by
    :func:`robot_motion.deploy.normalization.normalization_stats_hash`.
    """

    ref = str(stats_ref).strip()
    digest = str(stats_hash).strip().lower()
    if not ref:
        raise NormalizationContractError("OpenPI norm_stats stats_ref must be non-empty.")
    if "#" in ref:
        raise NormalizationContractError(
            "OpenPI norm_stats stats_ref must name the file without a #selector."
        )
    if _SHA256_RE.fullmatch(digest) is None:
        raise NormalizationContractError(
            "OpenPI norm_stats stats_hash must be a canonical sha256:<64-hex> value."
        )

    contract = NormalizationContract(
        state=NormalizationDomainSpec(
            owner="server",
            method="quantile_range",
            operation="normalize",
            input_space=state_input_space,
            output_space=state_output_space,
            stats_ref=f"{ref}#state",
            stats_hash=digest,
        ),
        action=NormalizationDomainSpec(
            owner="server",
            method="quantile_range",
            operation="unnormalize",
            input_space=action_input_space,
            output_space=action_output_space,
            stats_ref=f"{ref}#actions",
            stats_hash=digest,
        ),
        image=NormalizationDomainSpec(
            owner="none",
            method="identity",
            input_space=image_space,
            output_space=image_space,
            operation="identity",
        ),
    )
    return contract.to_dict()


__all__ = ["make_openpi_quantile_normalization_contract"]
