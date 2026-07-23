"""Validated readers for sealed v2.4 native-stream episode fragments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import (
    canonical_content_hash,
    canonical_loads,
)
from dataset_contracts.stream_catalog import (
    NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1,
    NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1,
    NATIVE_STREAM_CONTRACT_SCHEMA_V1,
    NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    NATIVE_STREAM_EPISODE_MANIFEST_SCHEMA_V1,
    NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1,
    NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
    NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1,
    NATIVE_STREAM_SAMPLE_SCHEMA_V1,
    NativeStreamContract,
    NativeStreamEpisodeManifest,
    NativeStreamEpisodeRange,
    NativeStreamSampleRecord,
)


class NativeStreamReadError(ValueError):
    """Raised when a sealed stream manifest or payload fails validation."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeStreamExternalPayloadReference:
    """Validated locator for payload bytes owned by a specialized artifact."""

    artifact_relative_path: str
    stored_payload_object_id: str
    payload_content_hash: str
    artifact_sha256: str
    artifact_byte_count: int
    encoding: str
    compression: str


@dataclass(frozen=True)
class NativeStreamEpisode:
    manifest: NativeStreamEpisodeManifest
    contracts_by_stream: Mapping[str, NativeStreamContract]
    records_by_stream: Mapping[str, tuple[NativeStreamSampleRecord, ...]]
    payload_by_stream_and_sample: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contracts_by_stream",
            MappingProxyType(dict(self.contracts_by_stream)),
        )
        object.__setattr__(
            self,
            "records_by_stream",
            MappingProxyType(dict(self.records_by_stream)),
        )
        object.__setattr__(
            self,
            "payload_by_stream_and_sample",
            MappingProxyType(
                {
                    stream_id: MappingProxyType(dict(payloads))
                    for stream_id, payloads in (
                        self.payload_by_stream_and_sample.items()
                    )
                }
            ),
        )


def _contract_from_payload(payload: object) -> NativeStreamContract:
    if not isinstance(payload, dict):
        raise NativeStreamReadError("Native stream contract is not a mapping.")
    if payload.get("schema") != NATIVE_STREAM_CONTRACT_SCHEMA_V1:
        raise NativeStreamReadError("Unsupported native stream contract schema.")
    try:
        contract = NativeStreamContract(
            stream_id=payload["stream_id"],
            modality=payload["modality"],
            producer_id=payload["producer_id"],
            value_schema=payload["value_schema"],
            shape=tuple(payload["shape"]),
            dtype=payload["dtype"],
            units=payload["units"],
            frame=payload["frame"],
            measured_clock_domain_id=payload["measured_clock_domain_id"],
            available_clock_domain_id=payload["available_clock_domain_id"],
            nominal_rate_hz=payload["nominal_rate_hz"],
            min_expected_rate_hz=payload["min_expected_rate_hz"],
            max_expected_rate_hz=payload["max_expected_rate_hz"],
            encoding=payload["encoding"],
            compression=payload["compression"],
            retention_policy=payload["retention_policy"],
            validity_policy=payload["validity_policy"],
            access_role=payload["access_role"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeStreamReadError(
            f"Invalid native stream contract: {exc}"
        ) from exc
    if contract.semantic_content_hash != payload.get("semantic_content_hash"):
        raise NativeStreamReadError("Native stream contract hash mismatch.")
    return contract


def _sample_from_payload(payload: object) -> NativeStreamSampleRecord:
    if not isinstance(payload, dict):
        raise NativeStreamReadError("Native stream sample is not a mapping.")
    if payload.get("schema") != NATIVE_STREAM_SAMPLE_SCHEMA_V1:
        raise NativeStreamReadError("Unsupported native stream sample schema.")
    try:
        record = NativeStreamSampleRecord(
            stream_id=payload["stream_id"],
            stream_contract_hash=payload["stream_contract_hash"],
            sample_id=payload["sample_id"],
            sequence=payload["sequence"],
            measured_at=payload["measured_at"],
            measured_clock_domain_id=payload["measured_clock_domain_id"],
            available_at=payload["available_at"],
            available_clock_domain_id=payload["available_clock_domain_id"],
            source_payload_object_id=payload["source_payload_object_id"],
            stored_payload_hash=payload["stored_payload_hash"],
            stored_payload_object_id=payload["stored_payload_object_id"],
            valid=payload["valid"],
            stale=payload["stale"],
            gap_reason=payload["gap_reason"],
            written_at=payload["written_at"],
            written_clock_domain_id=payload["written_clock_domain_id"],
            metadata=payload["metadata"],
            external_artifact_relative_path=payload.get(
                "external_artifact_relative_path"
            ),
            external_artifact_sha256=payload.get("external_artifact_sha256"),
            external_artifact_byte_count=payload.get(
                "external_artifact_byte_count"
            ),
            external_artifact_receipt_hash=payload.get(
                "external_artifact_receipt_hash"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeStreamReadError(f"Invalid native stream sample: {exc}") from exc
    if record.semantic_content_hash != payload.get("semantic_content_hash"):
        raise NativeStreamReadError("Native stream sample hash mismatch.")
    return record


def _range_from_payload(payload: object) -> NativeStreamEpisodeRange:
    if not isinstance(payload, dict):
        raise NativeStreamReadError("Native stream range is not a mapping.")
    try:
        return NativeStreamEpisodeRange(
            stream_id=payload["stream_id"],
            stream_contract_hash=payload["stream_contract_hash"],
            index_relative_path=payload["index_relative_path"],
            index_sha256=payload["index_sha256"],
            sample_count=payload["sample_count"],
            first_sequence=payload["first_sequence"],
            last_sequence=payload["last_sequence"],
            first_measured_at=payload["first_measured_at"],
            last_measured_at=payload["last_measured_at"],
            first_available_at=payload["first_available_at"],
            last_available_at=payload["last_available_at"],
            dropped_count=payload["dropped_count"],
            gap_reasons=tuple(payload["gap_reasons"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeStreamReadError(f"Invalid native stream range: {exc}") from exc


def _manifest_from_payload(payload: object) -> NativeStreamEpisodeManifest:
    if not isinstance(payload, dict):
        raise NativeStreamReadError("Native stream episode manifest is not a mapping.")
    if (
        payload.get("schema") != NATIVE_STREAM_EPISODE_MANIFEST_SCHEMA_V1
        or payload.get("status") != "sealed"
    ):
        raise NativeStreamReadError("Native stream episode is not a sealed v1 manifest.")
    try:
        manifest = NativeStreamEpisodeManifest(
            episode_uuid=payload["episode_uuid"],
            legacy_episode_index=payload["legacy_episode_index"],
            stream_ranges=tuple(
                _range_from_payload(value)
                for value in payload["stream_ranges"]
            ),
            sealed_at=payload["sealed_at"],
            sealed_clock_domain_id=payload["sealed_clock_domain_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeStreamReadError(
            f"Invalid native stream episode manifest: {exc}"
        ) from exc
    if manifest.semantic_content_hash != payload.get("semantic_content_hash"):
        raise NativeStreamReadError("Native stream episode manifest hash mismatch.")
    return manifest


def load_native_stream_episode(
    dataset_root: str | Path,
    episode_uuid: str,
    *,
    load_payloads: bool = True,
) -> NativeStreamEpisode:
    root = Path(dataset_root)
    manifest_path = (
        root
        / NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1
        / str(episode_uuid)
        / "manifest.cjson"
    )
    if not manifest_path.is_file():
        raise NativeStreamReadError(
            f"Native stream episode manifest does not exist: {manifest_path}"
        )
    manifest = _manifest_from_payload(
        canonical_loads(
            manifest_path.read_bytes(),
            expected_domain=NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    if manifest.episode_uuid != str(episode_uuid):
        raise NativeStreamReadError("Requested episode UUID does not match manifest.")

    contracts: dict[str, NativeStreamContract] = {}
    records_by_stream: dict[str, tuple[NativeStreamSampleRecord, ...]] = {}
    payloads_by_stream: dict[str, Mapping[str, Any]] = {}
    verified_external_artifacts: set[tuple[str, str, int]] = set()
    for stream_range in manifest.stream_ranges:
        contract_path = (
            root
            / NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1
            / f"{stream_range.stream_id}.cjson"
        )
        if not contract_path.is_file():
            raise NativeStreamReadError(
                f"Missing native stream contract for {stream_range.stream_id!r}."
            )
        contract = _contract_from_payload(
            canonical_loads(
                contract_path.read_bytes(),
                expected_domain=NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1,
            )
        )
        if contract.semantic_content_hash != stream_range.stream_contract_hash:
            raise NativeStreamReadError(
                "Native stream range and contract hash disagree."
            )
        index_path = root / stream_range.index_relative_path
        try:
            index_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise NativeStreamReadError(
                "Native stream index escapes the dataset root."
            ) from exc
        raw_index = index_path.read_bytes()
        if hashlib.sha256(raw_index).hexdigest() != stream_range.index_sha256:
            raise NativeStreamReadError("Native stream index hash mismatch.")
        records = tuple(
            _sample_from_payload(
                canonical_loads(
                    line,
                    expected_domain=NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1,
                )
            )
            for line in raw_index.splitlines()
            if line
        )
        if len(records) != stream_range.sample_count:
            raise NativeStreamReadError("Native stream sample count mismatch.")
        if any(
            record.stream_id != contract.stream_id
            or record.stream_contract_hash != contract.semantic_content_hash
            for record in records
        ):
            raise NativeStreamReadError(
                "Native stream sample does not match its contract."
            )
        if records:
            if (
                records[0].sequence != stream_range.first_sequence
                or records[-1].sequence != stream_range.last_sequence
            ):
                raise NativeStreamReadError(
                    "Native stream sequence range does not match the index."
                )
        payloads: dict[str, Any] = {}
        for record in records:
            if record.external_artifact_relative_path is not None:
                payload_path = root / record.external_artifact_relative_path
                try:
                    payload_path.resolve().relative_to(root.resolve())
                except ValueError as exc:
                    raise NativeStreamReadError(
                        "External native stream artifact escapes the dataset root."
                    ) from exc
                if not payload_path.is_file():
                    raise NativeStreamReadError(
                        "External native stream artifact is missing."
                    )
                artifact_byte_count = int(
                    record.external_artifact_byte_count
                )
                if payload_path.stat().st_size != artifact_byte_count:
                    raise NativeStreamReadError(
                        "External native stream artifact byte count mismatch."
                    )
                artifact_key = (
                    record.external_artifact_relative_path,
                    record.external_artifact_sha256,
                    artifact_byte_count,
                )
                if artifact_key not in verified_external_artifacts:
                    if (
                        _file_sha256(payload_path)
                        != record.external_artifact_sha256
                    ):
                        raise NativeStreamReadError(
                            "External native stream artifact hash mismatch."
                        )
                    verified_external_artifacts.add(artifact_key)
                if load_payloads:
                    payloads[record.sample_id] = (
                        NativeStreamExternalPayloadReference(
                            artifact_relative_path=(
                                record.external_artifact_relative_path
                            ),
                            stored_payload_object_id=(
                                record.stored_payload_object_id
                            ),
                            payload_content_hash=record.stored_payload_hash,
                            artifact_sha256=record.external_artifact_sha256,
                            artifact_byte_count=artifact_byte_count,
                            encoding=contract.encoding,
                            compression=contract.compression,
                        )
                    )
                continue
            if load_payloads:
                payload_path = (
                    index_path.parent
                    / "payloads"
                    / f"{record.stored_payload_hash}.cjson"
                )
                payload = canonical_loads(
                    payload_path.read_bytes(),
                    expected_domain=NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
                )
                if (
                    canonical_content_hash(
                        payload,
                        domain=NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
                    )
                    != record.stored_payload_hash
                ):
                    raise NativeStreamReadError(
                        "Native stream payload hash mismatch."
                    )
                payloads[record.sample_id] = payload
        contracts[contract.stream_id] = contract
        records_by_stream[contract.stream_id] = records
        payloads_by_stream[contract.stream_id] = MappingProxyType(payloads)
    return NativeStreamEpisode(
        manifest=manifest,
        contracts_by_stream=contracts,
        records_by_stream=records_by_stream,
        payload_by_stream_and_sample=payloads_by_stream,
    )


__all__ = [
    "NativeStreamEpisode",
    "NativeStreamExternalPayloadReference",
    "NativeStreamReadError",
    "load_native_stream_episode",
]
