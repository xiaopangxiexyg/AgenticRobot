"""Versioned, checksummed and process-safe MJB artifact cache."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from project_paths import repo_path, resolve_path
from sim.contracts import SimulationContract


MODEL_CACHE_ENV = "SIM_MUJOCO_CACHE_DIR"
ARTIFACT_KEY_VERSION = "dexteleop.sim.mujoco.model_artifact.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_dependencies(contract: SimulationContract) -> tuple[Path, ...]:
    from ..urdf import parse_urdf

    urdf_path = resolve_path(contract.scene.urdf_path)
    model = parse_urdf(urdf_path)
    dependencies: set[Path] = {urdf_path, resolve_path(contract.scene.camera_registry_path)}
    for link in model.links.values():
        geometries = [item.geometry for item in link.visuals]
        geometries.extend(item.geometry for item in link.collisions)
        for geometry in geometries:
            if geometry.filename is None:
                continue
            mesh_path = model.resolve_mesh_path(geometry.filename)
            dependencies.add(mesh_path)
            if mesh_path.suffix.lower() == ".obj":
                try:
                    lines = mesh_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for line in lines:
                    if line.lstrip().startswith("mtllib "):
                        relative = line.strip().split(maxsplit=1)[1]
                        material = (mesh_path.parent / relative).resolve()
                        if material.exists():
                            dependencies.add(material)
    return tuple(sorted(dependencies))


@dataclass(frozen=True)
class ModelArtifact:
    key: str
    directory: Path
    mjb_path: Path
    mjcf_path: Path
    manifest_path: Path
    cache_hit: bool


class ModelArtifactCache:
    """Compile once and load a standalone MJB on subsequent constructions."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = os.environ.get(MODEL_CACHE_ENV, "").strip()
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else Path(configured).expanduser().resolve()
            if configured
            else repo_path(".cache", "mujoco")
        )

    def _fingerprint(
        self,
        contract: SimulationContract,
        *,
        position_actuation_profile=None,
        gripper_actuation_profile=None,
    ) -> tuple[str, dict[str, object]]:
        import mujoco

        dependency_hashes = {
            str(path): _sha256(path) for path in _mesh_dependencies(contract)
        }
        payload: dict[str, object] = {
            "artifact_key_version": ARTIFACT_KEY_VERSION,
            "contract": contract.to_dict(),
            "position_actuation_profile": (
                None
                if position_actuation_profile is None
                else position_actuation_profile.to_metadata()
            ),
            "gripper_actuation_profile": (
                None
                if gripper_actuation_profile is None
                else gripper_actuation_profile.to_metadata()
            ),
            "dependencies": dependency_hashes,
            "mujoco_python_version": mujoco.__version__,
            "mujoco_native_version": mujoco.mj_versionString(),
            "platform": platform.system(),
            "machine": platform.machine(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), payload

    def load_scene(
        self,
        contract: SimulationContract,
        *,
        position_actuation_profile=None,
        gripper_actuation_profile=None,
    ):
        """Return ``(scene, artifact)`` after validating or building the cache."""

        import mujoco

        from ..scene import MujocoScene

        key, fingerprint = self._fingerprint(
            contract,
            position_actuation_profile=position_actuation_profile,
            gripper_actuation_profile=gripper_actuation_profile,
        )
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        mjb_path = directory / "model.mjb"
        mjcf_path = directory / "model.xml"
        manifest_path = directory / "manifest.json"
        lock_path = directory / ".build.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._valid(manifest_path, mjb_path, mjcf_path, key):
                try:
                    model = mujoco.MjModel.from_binary_path(str(mjb_path))
                    scene = MujocoScene.from_compiled_contract(
                        contract,
                        model=model,
                        mjcf_path=mjcf_path,
                    )
                except Exception:
                    manifest_path.unlink(missing_ok=True)
                    mjb_path.unlink(missing_ok=True)
                else:
                    return scene, ModelArtifact(
                        key, directory, mjb_path, mjcf_path, manifest_path, True
                    )

            scene = MujocoScene.from_contract(
                contract,
                mjcf_path=mjcf_path,
                position_actuation_profile=position_actuation_profile,
                gripper_actuation_profile=gripper_actuation_profile,
            )
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix="model.",
                suffix=".mjb.tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                mujoco.mj_saveModel(scene.model, str(temporary_path))
                os.replace(temporary_path, mjb_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            manifest = {
                "schema_version": ARTIFACT_KEY_VERSION,
                "artifact_key": key,
                "fingerprint": fingerprint,
                "mjb_sha256": _sha256(mjb_path),
                "mjcf_sha256": _sha256(mjcf_path),
            }
            self._atomic_json(manifest_path, manifest)
            return scene, ModelArtifact(key, directory, mjb_path, mjcf_path, manifest_path, False)

    def clear(self, key: str | None = None) -> None:
        """Explicitly remove one artifact or the complete configured cache."""

        if key is None:
            if not self.root.is_dir():
                return
            for child in self.root.iterdir():
                if child.is_dir() and len(child.name) == 64:
                    self.clear(child.name)
            return
        normalized = str(key).lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("artifact cache key must be a SHA-256 hex digest.")
        target = self.root / normalized
        if target.is_dir():
            shutil.rmtree(target)

    @staticmethod
    def _valid(manifest_path: Path, mjb_path: Path, mjcf_path: Path, key: str) -> bool:
        if not manifest_path.is_file() or not mjb_path.is_file() or not mjcf_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return bool(
                manifest.get("schema_version") == ARTIFACT_KEY_VERSION
                and manifest.get("artifact_key") == key
                and manifest.get("mjb_sha256") == _sha256(mjb_path)
                and manifest.get("mjcf_sha256") == _sha256(mjcf_path)
            )
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, object]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["MODEL_CACHE_ENV", "ModelArtifact", "ModelArtifactCache"]
