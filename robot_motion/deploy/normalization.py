"""Compatibility exports for the pure normalization contracts.

The schema and validation ownership lives in :mod:`dataset_contracts.normalization`.
Keep this module so existing robot-motion callers do not need a flag-day import
migration.
"""

from dataset_contracts.normalization import (
    LEGACY_NORMALIZATION_SCHEMAS,
    NORMALIZATION_CONTRACT_HASH_DOMAIN_V1,
    NORMALIZATION_CONTRACT_SCHEMA_V1,
    NORMALIZATION_DOMAINS,
    NORMALIZATION_METHODS,
    NORMALIZATION_OPERATIONS,
    NORMALIZATION_OWNERS,
    NormalizationContract,
    NormalizationContractError,
    NormalizationDomainSpec,
    explicit_identity_normalization_contract,
    normalization_contract_hash,
    normalization_stats_hash,
)

__all__ = [
    "LEGACY_NORMALIZATION_SCHEMAS",
    "NORMALIZATION_CONTRACT_HASH_DOMAIN_V1",
    "NORMALIZATION_CONTRACT_SCHEMA_V1",
    "NORMALIZATION_DOMAINS",
    "NORMALIZATION_METHODS",
    "NORMALIZATION_OPERATIONS",
    "NORMALIZATION_OWNERS",
    "NormalizationContract",
    "NormalizationContractError",
    "NormalizationDomainSpec",
    "explicit_identity_normalization_contract",
    "normalization_contract_hash",
    "normalization_stats_hash",
]
