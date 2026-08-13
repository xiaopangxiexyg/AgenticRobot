from __future__ import annotations

import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


Vec3 = tuple[float, float, float]
Mat4 = tuple[tuple[float, float, float, float], ...]


class URDFValidationError(ValueError):
    """Raised when a URDF cannot be edited or exported safely."""


@dataclass(frozen=True)
class GeometryRef:
    link_name: str
    role: str
    index: int
    element: ET.Element

    @property
    def label(self) -> str:
        name = self.element.get("name", "unnamed")
        return f"{self.role}[{self.index}] {name} ({geometry_kind(self.element)})"


@dataclass(frozen=True)
class Endpoint:
    driving_joint: str
    root_link: str
    link_name: str
    geometry: GeometryRef


@dataclass(frozen=True)
class GripperPair:
    joint_a: str
    joint_b: str
    parent_link: str
    score: float
    reason: str

    @property
    def label(self) -> str:
        return f"{self.parent_link}: {self.joint_a} + {self.joint_b}"


@dataclass(frozen=True)
class PoseTarget:
    element: ET.Element
    label: str
    owner_name: str


@dataclass(frozen=True)
class ValidationReport:
    warnings: tuple[str, ...]


def _floats(text: str | None, count: int, default: Sequence[float]) -> tuple[float, ...]:
    if not text:
        return tuple(default)
    try:
        values = tuple(float(value) for value in text.split())
    except ValueError as exc:
        raise URDFValidationError(f"Invalid numeric vector: {text!r}") from exc
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise URDFValidationError(f"Expected {count} finite values, got {text!r}")
    return values


def parse_xyz(element: ET.Element | None) -> Vec3:
    if element is None:
        return (0.0, 0.0, 0.0)
    return _floats(element.get("xyz"), 3, (0.0, 0.0, 0.0))  # type: ignore[return-value]


def parse_rpy(element: ET.Element | None) -> Vec3:
    if element is None:
        return (0.0, 0.0, 0.0)
    return _floats(element.get("rpy"), 3, (0.0, 0.0, 0.0))  # type: ignore[return-value]


def format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _origin(owner: ET.Element, *, create: bool = False) -> ET.Element | None:
    origin = owner.find("origin")
    if origin is None and create:
        origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        owner.insert(0, origin)
    return origin


def get_origin(owner: ET.Element) -> tuple[Vec3, Vec3]:
    origin = _origin(owner)
    return parse_xyz(origin), parse_rpy(origin)


def set_origin(owner: ET.Element, xyz: Sequence[float], rpy: Sequence[float]) -> None:
    if len(xyz) != 3 or len(rpy) != 3:
        raise URDFValidationError("Origin requires three xyz and three rpy values")
    values = tuple(float(v) for v in (*xyz, *rpy))
    if not all(math.isfinite(v) for v in values):
        raise URDFValidationError("Origin values must be finite")
    origin = _origin(owner, create=True)
    assert origin is not None
    origin.set("xyz", format_vector(values[:3]))
    origin.set("rpy", format_vector(values[3:]))


def identity() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def matmul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4))
        for row in range(4)
    )


def transform_point(matrix: Mat4, point: Sequence[float]) -> Vec3:
    return tuple(
        sum(matrix[row][col] * float(point[col]) for col in range(3)) + matrix[row][3]
        for row in range(3)
    )  # type: ignore[return-value]


def transform_vector(matrix: Mat4, vector: Sequence[float]) -> Vec3:
    return tuple(
        sum(matrix[row][col] * float(vector[col]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def invert_rigid(matrix: Mat4) -> Mat4:
    rotation_t = tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))
    translation = tuple(matrix[row][3] for row in range(3))
    inv_translation = tuple(-sum(rotation_t[row][k] * translation[k] for k in range(3)) for row in range(3))
    return tuple(
        tuple(rotation_t[row][col] if col < 3 else inv_translation[row] for col in range(4))
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> Mat4:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses fixed-axis roll/pitch/yaw, equivalent to Rz(yaw) Ry(pitch) Rx(roll).
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return tuple(
        tuple(rotation[row][col] if col < 3 else float(xyz[row]) for col in range(4))
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _axis_angle(axis: Sequence[float], angle: float) -> Mat4:
    x, y, z = normalized(axis)
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )
    return tuple(
        tuple(rotation[row][col] if col < 3 else 0.0 for col in range(4))
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _translation(axis: Sequence[float], distance: float) -> Mat4:
    x, y, z = normalized(axis)
    return origin_matrix((x * distance, y * distance, z * distance), (0.0, 0.0, 0.0))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def normalized(vector: Sequence[float]) -> Vec3:
    length = norm(vector)
    if length <= 1e-12:
        raise URDFValidationError("Joint axis cannot be zero")
    return tuple(float(value) / length for value in vector)  # type: ignore[return-value]


def geometry_kind(owner: ET.Element) -> str:
    geometry = owner.find("geometry")
    if geometry is None or len(geometry) != 1:
        return "unknown"
    return geometry[0].tag


class URDFDocument:
    """Mutable URDF DOM plus gripper-specific detection and editing helpers."""

    def __init__(self, source_path: Path, tree: ET.ElementTree):
        self.source_path = source_path.resolve()
        self.tree = tree
        self.root = tree.getroot()
        self._reindex()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "URDFDocument":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise URDFValidationError(f"URDF does not exist: {source}")
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        try:
            tree = ET.parse(source, parser=parser)
        except (ET.ParseError, OSError) as exc:
            raise URDFValidationError(f"Cannot parse URDF {source}: {exc}") from exc
        if tree.getroot().tag != "robot":
            raise URDFValidationError("Expected a <robot> document root")
        document = cls(source, tree)
        document.validate()
        return document

    def _reindex(self) -> None:
        self.links = {
            element.get("name", ""): element
            for element in self.root.findall("link")
            if element.get("name")
        }
        self.joints = {
            element.get("name", ""): element
            for element in self.root.findall("joint")
            if element.get("name")
        }
        self.parent_joint_by_child: dict[str, ET.Element] = {}
        self.joints_by_parent: dict[str, list[ET.Element]] = {}
        for joint in self.joints.values():
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name, child_name = parent.get("link", ""), child.get("link", "")
            if child_name:
                self.parent_joint_by_child[child_name] = joint
            self.joints_by_parent.setdefault(parent_name, []).append(joint)

    def snapshot(self) -> bytes:
        return ET.tostring(self.root, encoding="utf-8")

    def restore(self, snapshot: bytes) -> None:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        self.root = ET.fromstring(snapshot, parser=parser)
        self.tree = ET.ElementTree(self.root)
        self._reindex()

    def validate(self) -> ValidationReport:
        warnings: list[str] = []
        link_elements = [element for element in self.root.findall("link")]
        joint_elements = [element for element in self.root.findall("joint")]
        link_names = [element.get("name", "") for element in link_elements]
        joint_names = [element.get("name", "") for element in joint_elements]
        if not link_elements:
            raise URDFValidationError("URDF contains no links")
        for kind, names in (("link", link_names), ("joint", joint_names)):
            missing = [index for index, name in enumerate(names) if not name]
            duplicates = sorted({name for name in names if name and names.count(name) > 1})
            if missing:
                raise URDFValidationError(f"Unnamed {kind} element(s): {missing}")
            if duplicates:
                raise URDFValidationError(f"Duplicate {kind} name(s): {', '.join(duplicates)}")
        link_set = set(link_names)
        for joint in joint_elements:
            name = joint.get("name", "<unnamed>")
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                raise URDFValidationError(f"Joint {name} must have parent and child")
            parent_name, child_name = parent.get("link", ""), child.get("link", "")
            if parent_name not in link_set or child_name not in link_set:
                raise URDFValidationError(
                    f"Joint {name} references missing link(s): {parent_name!r}, {child_name!r}"
                )
            if joint.get("type") in {"prismatic", "revolute"}:
                normalized(parse_xyz(joint.find("axis")) if joint.find("axis") is not None else (1.0, 0.0, 0.0))
                limit = joint.find("limit")
                if limit is None:
                    raise URDFValidationError(f"Joint {name} has no limit")
                try:
                    lower, upper = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
                except ValueError as exc:
                    raise URDFValidationError(f"Joint {name} has invalid limits") from exc
                if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                    raise URDFValidationError(f"Joint {name} requires finite lower <= upper limits")
            mimic = joint.find("mimic")
            if mimic is not None and mimic.get("joint", "") not in set(joint_names):
                raise URDFValidationError(f"Joint {name} mimics missing joint {mimic.get('joint')!r}")

        # A link can have only one parent in a URDF tree; catch loops as well.
        parent_by_child: dict[str, str] = {}
        for joint in joint_elements:
            parent_name = joint.find("parent").get("link", "")  # type: ignore[union-attr]
            child_name = joint.find("child").get("link", "")  # type: ignore[union-attr]
            if child_name in parent_by_child:
                raise URDFValidationError(f"Link {child_name} has multiple parent joints")
            parent_by_child[child_name] = parent_name
        for link_name in link_names:
            seen: set[str] = set()
            cursor = link_name
            while cursor in parent_by_child:
                if cursor in seen:
                    raise URDFValidationError(f"Kinematic cycle includes link {cursor}")
                seen.add(cursor)
                cursor = parent_by_child[cursor]
        roots = [name for name in link_names if name not in parent_by_child]
        if len(roots) > 1:
            warnings.append(f"URDF has {len(roots)} disconnected root links")
        return ValidationReport(tuple(warnings))

    def joint_parent(self, joint: ET.Element | str) -> str:
        element = self.joints[joint] if isinstance(joint, str) else joint
        parent = element.find("parent")
        return "" if parent is None else parent.get("link", "")

    def joint_child(self, joint: ET.Element | str) -> str:
        element = self.joints[joint] if isinstance(joint, str) else joint
        child = element.find("child")
        return "" if child is None else child.get("link", "")

    def joint_axis(self, joint: ET.Element | str) -> Vec3:
        element = self.joints[joint] if isinstance(joint, str) else joint
        axis = element.find("axis")
        return normalized(parse_xyz(axis) if axis is not None else (1.0, 0.0, 0.0))

    def joint_limits(self, joint_name: str) -> tuple[float, float]:
        limit = self.joints[joint_name].find("limit")
        if limit is None:
            return (0.0, 0.0)
        return float(limit.get("lower", "0")), float(limit.get("upper", "0"))

    def _mimic(self, joint_name: str) -> tuple[str, float, float] | None:
        mimic = self.joints[joint_name].find("mimic")
        if mimic is None or not mimic.get("joint"):
            return None
        return (
            mimic.get("joint", ""),
            float(mimic.get("multiplier", "1")),
            float(mimic.get("offset", "0")),
        )

    def detect_gripper_pairs(self) -> list[GripperPair]:
        prismatic = [joint for joint in self.joints.values() if joint.get("type") == "prismatic"]
        pairs: list[GripperPair] = []
        for index, first in enumerate(prismatic):
            for second in prismatic[index + 1 :]:
                name_a, name_b = first.get("name", ""), second.get("name", "")
                parent_a, parent_b = self.joint_parent(first), self.joint_parent(second)
                mimic_a, mimic_b = self._mimic(name_a), self._mimic(name_b)
                linked = bool(
                    (mimic_a and mimic_a[0] == name_b)
                    or (mimic_b and mimic_b[0] == name_a)
                    or parent_a == parent_b
                )
                if not linked:
                    continue
                xyz_a, rpy_a = get_origin(first)
                xyz_b, rpy_b = get_origin(second)
                axis_a = transform_vector(origin_matrix((0, 0, 0), rpy_a), self.joint_axis(first))
                axis_b = transform_vector(origin_matrix((0, 0, 0), rpy_b), self.joint_axis(second))
                alignment = abs(dot(normalized(axis_a), normalized(axis_b)))
                if alignment < 0.92:
                    continue
                score = alignment * 4.0
                reasons = [f"parallel axes {alignment:.3f}"]
                if parent_a == parent_b:
                    score += 6.0
                    reasons.append("shared parent")
                if (mimic_a and mimic_a[0] == name_b) or (mimic_b and mimic_b[0] == name_a):
                    score += 8.0
                    reasons.append("mimic coupled")
                words = f"{name_a} {name_b} {parent_a} {parent_b}".lower()
                for token, bonus in (("gripper", 2.0), ("finger", 2.0), ("jaw", 2.0)):
                    if token in words:
                        score += bonus
                separation = norm(tuple(x - y for x, y in zip(xyz_a, xyz_b)))
                if separation > 1e-4:
                    score += min(2.0, separation * 20.0)
                parent = parent_a if parent_a == parent_b else self.common_ancestor_link(parent_a, parent_b)
                if not parent:
                    continue
                pairs.append(GripperPair(name_a, name_b, parent, score, ", ".join(reasons)))
        return sorted(pairs, key=lambda pair: (-pair.score, pair.label))

    def common_ancestor_link(self, link_a: str, link_b: str) -> str | None:
        ancestors_b = {link_b, *self.ancestor_links(link_b)}
        for candidate in (link_a, *self.ancestor_links(link_a)):
            if candidate in ancestors_b:
                return candidate
        return None

    def subtree_links(self, root_link: str, *, fixed_only: bool = False) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        queue = [(root_link, 0)]
        seen: set[str] = set()
        while queue:
            link, depth = queue.pop(0)
            if link in seen:
                continue
            seen.add(link)
            result.append((link, depth))
            for joint in self.joints_by_parent.get(link, []):
                if fixed_only and joint.get("type") != "fixed":
                    continue
                queue.append((self.joint_child(joint), depth + 1))
        return result

    def geometry_refs(self, link_name: str) -> list[GeometryRef]:
        link = self.links[link_name]
        refs: list[GeometryRef] = []
        for role in ("visual", "collision"):
            for index, element in enumerate(link.findall(role)):
                refs.append(GeometryRef(link_name, role, index, element))
        return refs

    def endpoint_candidates(self, joint_name: str) -> list[Endpoint]:
        root_link = self.joint_child(joint_name)
        candidates: list[tuple[float, Endpoint]] = []
        for link_name, depth in self.subtree_links(root_link, fixed_only=True):
            lowered = link_name.lower()
            name_score = 0.0
            for token, bonus in (
                ("end_component", 22.0),
                ("fingertip", 20.0),
                ("finger_tip", 20.0),
                ("terminal", 18.0),
                ("pad", 14.0),
                ("finger", 10.0),
                ("jaw", 10.0),
                ("contact", 3.0),
                ("linkage", -18.0),
                ("carriage", -10.0),
            ):
                if token in lowered:
                    name_score += bonus
            for ref in self.geometry_refs(link_name):
                kind = geometry_kind(ref.element)
                role_score = 4.0 if ref.role == "visual" else 2.0
                primitive_score = 4.0 if kind in {"box", "cylinder", "sphere"} else 0.0
                score = name_score + role_score + primitive_score + min(depth, 3) * 0.5
                candidates.append((score, Endpoint(joint_name, root_link, link_name, ref)))
        candidates.sort(key=lambda item: (-item[0], item[1].geometry.label))
        return [endpoint for _, endpoint in candidates]

    def default_endpoint(self, joint_name: str) -> Endpoint:
        candidates = self.endpoint_candidates(joint_name)
        if not candidates:
            raise URDFValidationError(f"No visual or collision geometry below joint {joint_name}")
        # Prefer an editable primitive on the best-scoring link. This selects the
        # pad collision for mesh-backed proxy jaws and the visual box for templates.
        top_link = candidates[0].link_name
        same_link = [candidate for candidate in candidates if candidate.link_name == top_link]
        primitives = [candidate for candidate in same_link if geometry_kind(candidate.geometry.element) in {"box", "cylinder", "sphere"}]
        visuals = [candidate for candidate in primitives if candidate.geometry.role == "visual"]
        if visuals:
            return visuals[0]
        if primitives:
            return primitives[0]
        return same_link[0]

    def endpoint_for_geometry(self, joint_name: str, link_name: str, role: str, index: int) -> Endpoint:
        refs = self.geometry_refs(link_name)
        for ref in refs:
            if ref.role == role and ref.index == index:
                return Endpoint(joint_name, self.joint_child(joint_name), link_name, ref)
        raise URDFValidationError(f"Geometry {link_name}/{role}[{index}] not found")

    def pose_target(self, endpoint: Endpoint) -> PoseTarget:
        parent_joint = self.parent_joint_by_child.get(endpoint.link_name)
        if parent_joint is not None and parent_joint.get("type") == "fixed" and endpoint.link_name != endpoint.root_link:
            return PoseTarget(parent_joint, "fixed mount origin", parent_joint.get("name", ""))
        return PoseTarget(endpoint.geometry.element, f"{endpoint.geometry.role} geometry origin", endpoint.geometry.label)

    def geometry_dimensions(
        self,
        geometry_ref: GeometryRef,
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
    ) -> Vec3 | None:
        geometry = geometry_ref.element.find("geometry")
        if geometry is None or len(geometry) != 1:
            return None
        shape = geometry[0]
        if shape.tag == "box":
            return _floats(shape.get("size"), 3, (1.0, 1.0, 1.0))  # type: ignore[return-value]
        if shape.tag == "cylinder":
            radius, length = float(shape.get("radius", "0")), float(shape.get("length", "0"))
            return (radius * 2.0, radius * 2.0, length)
        if shape.tag == "sphere":
            diameter = float(shape.get("radius", "0")) * 2.0
            return (diameter, diameter, diameter)
        if shape.tag == "mesh" and mesh_intrinsic_dimensions is not None:
            mesh_path = self.resolve_mesh_filename(shape.get("filename", ""))
            if mesh_path is None:
                return None
            intrinsic = mesh_intrinsic_dimensions(mesh_path)
            scale = _floats(shape.get("scale"), 3, (1.0, 1.0, 1.0))
            return tuple(abs(intrinsic[index] * scale[index]) for index in range(3))  # type: ignore[return-value]
        return None

    def set_geometry_dimensions(
        self,
        geometry_ref: GeometryRef,
        dimensions: Sequence[float],
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
        mesh_intrinsic_center: Callable[[Path], Vec3] | None = None,
    ) -> None:
        dims = tuple(float(value) for value in dimensions)
        if len(dims) != 3 or not all(math.isfinite(value) and value > 0 for value in dims):
            raise URDFValidationError("Geometry dimensions must be three positive finite values")
        geometry = geometry_ref.element.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise URDFValidationError(f"Geometry is malformed: {geometry_ref.label}")
        shape = geometry[0]
        if shape.tag == "box":
            shape.set("size", format_vector(dims))
        elif shape.tag == "cylinder":
            if not math.isclose(dims[0], dims[1], rel_tol=1e-5, abs_tol=1e-9):
                raise URDFValidationError("Cylinder X and Y dimensions must match")
            shape.set("radius", f"{dims[0] / 2.0:.9g}")
            shape.set("length", f"{dims[2]:.9g}")
        elif shape.tag == "sphere":
            if max(dims) - min(dims) > max(dims) * 1e-5:
                raise URDFValidationError("Sphere X, Y and Z dimensions must match")
            shape.set("radius", f"{dims[0] / 2.0:.9g}")
        elif shape.tag == "mesh":
            if mesh_intrinsic_dimensions is None:
                raise URDFValidationError("Mesh sizing requires readable mesh bounds")
            mesh_path = self.resolve_mesh_filename(shape.get("filename", ""))
            if mesh_path is None:
                raise URDFValidationError(f"Cannot resolve mesh: {shape.get('filename', '')}")
            intrinsic = mesh_intrinsic_dimensions(mesh_path)
            if any(value <= 1e-12 for value in intrinsic):
                raise URDFValidationError(f"Mesh has a degenerate bounding box: {mesh_path}")
            old_scale = _floats(shape.get("scale"), 3, (1.0, 1.0, 1.0))
            new_scale = tuple(dims[i] / intrinsic[i] for i in range(3))
            if mesh_intrinsic_center is not None:
                center = mesh_intrinsic_center(mesh_path)
                xyz, rpy = get_origin(geometry_ref.element)
                rotation = origin_matrix((0.0, 0.0, 0.0), rpy)
                local_shift = tuple(center[i] * (old_scale[i] - new_scale[i]) for i in range(3))
                world_shift = transform_vector(rotation, local_shift)
                set_origin(
                    geometry_ref.element,
                    tuple(xyz[i] + world_shift[i] for i in range(3)),
                    rpy,
                )
            shape.set("scale", format_vector(new_scale))
        else:
            raise URDFValidationError(f"Unsupported geometry type: {shape.tag}")

    def matching_geometry(
        self,
        geometry_ref: GeometryRef,
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
    ) -> GeometryRef | None:
        kind = geometry_kind(geometry_ref.element)
        other_role = "collision" if geometry_ref.role == "visual" else "visual"
        candidates = [ref for ref in self.geometry_refs(geometry_ref.link_name) if ref.role == other_role]
        same_kind = [ref for ref in candidates if geometry_kind(ref.element) == kind]
        if same_kind:
            return same_kind[0]
        source_dimensions = self.geometry_dimensions(geometry_ref, mesh_intrinsic_dimensions)
        if source_dimensions is None:
            return None
        compatible: list[tuple[float, GeometryRef]] = []
        for candidate in candidates:
            dimensions = self.geometry_dimensions(candidate, mesh_intrinsic_dimensions)
            if dimensions is None:
                continue
            relative_error = max(
                abs(dimensions[index] - source_dimensions[index]) / max(source_dimensions[index], 1e-12)
                for index in range(3)
            )
            if relative_error <= 0.05:
                compatible.append((relative_error, candidate))
        return min(compatible, default=(math.inf, None), key=lambda item: item[0])[1]

    def master_joint(self, pair: GripperPair) -> str:
        mimic_a, mimic_b = self._mimic(pair.joint_a), self._mimic(pair.joint_b)
        if mimic_a and mimic_a[0] == pair.joint_b:
            return pair.joint_b
        if mimic_b and mimic_b[0] == pair.joint_a:
            return pair.joint_a
        return pair.joint_a

    def pair_joint_positions(self, pair: GripperPair, master_q: float) -> dict[str, float]:
        master = self.master_joint(pair)
        positions = {master: float(master_q)}
        other = pair.joint_b if master == pair.joint_a else pair.joint_a
        mimic = self._mimic(other)
        if mimic and mimic[0] == master:
            positions[other] = mimic[1] * float(master_q) + mimic[2]
        else:
            positions[other] = float(master_q)
        return positions

    def set_pair_limits(self, pair: GripperPair, closed_q: float, open_q: float) -> None:
        values = (float(closed_q), float(open_q))
        if not all(math.isfinite(value) for value in values) or math.isclose(values[0], values[1], abs_tol=1e-12):
            raise URDFValidationError("Open and closed joint positions must be distinct finite values")
        master = self.master_joint(pair)
        low, high = sorted(values)
        master_limit = self.joints[master].find("limit")
        assert master_limit is not None
        master_limit.set("lower", f"{low:.9g}")
        master_limit.set("upper", f"{high:.9g}")
        other = pair.joint_b if master == pair.joint_a else pair.joint_a
        other_limit = self.joints[other].find("limit")
        if other_limit is not None:
            mimic = self._mimic(other)
            transformed = values if not mimic else tuple(mimic[1] * value + mimic[2] for value in values)
            other_limit.set("lower", f"{min(transformed):.9g}")
            other_limit.set("upper", f"{max(transformed):.9g}")

    def joint_motion_matrix(self, joint: ET.Element, position: float) -> Mat4:
        xyz, rpy = get_origin(joint)
        base = origin_matrix(xyz, rpy)
        joint_type = joint.get("type", "fixed")
        if joint_type == "prismatic":
            return matmul(base, _translation(self.joint_axis(joint), position))
        if joint_type in {"revolute", "continuous"}:
            return matmul(base, _axis_angle(self.joint_axis(joint), position))
        return base

    def link_transforms(self, joint_positions: dict[str, float] | None = None) -> dict[str, Mat4]:
        positions = joint_positions or {}
        transforms: dict[str, Mat4] = {}
        roots = [name for name in self.links if name not in self.parent_joint_by_child]
        queue = [(root, identity()) for root in roots]
        while queue:
            link_name, link_transform = queue.pop(0)
            if link_name in transforms:
                continue
            transforms[link_name] = link_transform
            for joint in self.joints_by_parent.get(link_name, []):
                name = joint.get("name", "")
                q = positions.get(name, 0.0)
                child_transform = matmul(link_transform, self.joint_motion_matrix(joint, q))
                queue.append((self.joint_child(joint), child_transform))
        return transforms

    def geometry_transform(self, geometry_ref: GeometryRef, link_transforms: dict[str, Mat4]) -> Mat4:
        xyz, rpy = get_origin(geometry_ref.element)
        return matmul(link_transforms[geometry_ref.link_name], origin_matrix(xyz, rpy))

    def contact_frame(self, endpoint: Endpoint) -> str | None:
        scored: list[tuple[int, str]] = []
        for link_name, depth in self.subtree_links(endpoint.root_link, fixed_only=True):
            lowered = link_name.lower()
            if "contact" in lowered and not self.geometry_refs(link_name):
                score = (10 if "inner" in lowered else 0) + (5 if endpoint.link_name in self.ancestor_links(link_name) else 0) - depth
                scored.append((score, link_name))
        return max(scored, default=(0, ""))[1] or None

    def snap_contact_frame_to_inner_face(
        self,
        pair: GripperPair,
        endpoint: Endpoint,
        other_endpoint: Endpoint,
        master_q: float,
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
    ) -> bool:
        """Place a direct fixed contact frame on the selected geometry's inward face.

        The dominant local axis toward the opposite jaw is used. This deliberately
        avoids guessing a contact point on an edge or corner when a jaw is slightly
        rotated relative to the opening direction.
        """
        contact_link = self.contact_frame(endpoint)
        if contact_link is None:
            return False
        contact_joint = self.parent_joint_by_child.get(contact_link)
        if (
            contact_joint is None
            or contact_joint.get("type") != "fixed"
            or self.joint_parent(contact_joint) != endpoint.link_name
        ):
            return False
        dimensions = self.geometry_dimensions(endpoint.geometry, mesh_intrinsic_dimensions)
        if dimensions is None:
            return False
        transforms = self.link_transforms(self.pair_joint_positions(pair, master_q))
        endpoint_geometry = self.geometry_transform(endpoint.geometry, transforms)
        other_geometry = self.geometry_transform(other_endpoint.geometry, transforms)
        center = transform_point(endpoint_geometry, (0.0, 0.0, 0.0))
        other_center = transform_point(other_geometry, (0.0, 0.0, 0.0))
        direction_world = tuple(other_center[index] - center[index] for index in range(3))
        if norm(direction_world) <= 1e-12:
            return False
        direction_local = transform_vector(invert_rigid(endpoint_geometry), direction_world)
        dominant = max(range(3), key=lambda index: abs(direction_local[index]))
        face_center = [0.0, 0.0, 0.0]
        face_center[dominant] = math.copysign(dimensions[dominant] / 2.0, direction_local[dominant])
        point_world = transform_point(endpoint_geometry, face_center)
        point_parent = transform_point(invert_rigid(transforms[endpoint.link_name]), point_world)
        _old_xyz, old_rpy = get_origin(contact_joint)
        set_origin(contact_joint, point_parent, old_rpy)
        return True

    def ancestor_links(self, link_name: str) -> list[str]:
        result: list[str] = []
        cursor = link_name
        while cursor in self.parent_joint_by_child:
            joint = self.parent_joint_by_child[cursor]
            cursor = self.joint_parent(joint)
            result.append(cursor)
        return result

    def opening_gap(
        self,
        pair: GripperPair,
        endpoint_a: Endpoint,
        endpoint_b: Endpoint,
        master_q: float,
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
    ) -> float | None:
        transforms = self.link_transforms(self.pair_joint_positions(pair, master_q))
        contact_a, contact_b = self.contact_frame(endpoint_a), self.contact_frame(endpoint_b)
        if contact_a and contact_b:
            matrix_a = self.geometry_transform(endpoint_a.geometry, transforms)
            matrix_b = self.geometry_transform(endpoint_b.geometry, transforms)
            center_a = transform_point(matrix_a, (0.0, 0.0, 0.0))
            center_b = transform_point(matrix_b, (0.0, 0.0, 0.0))
            center_delta = tuple(center_a[index] - center_b[index] for index in range(3))
            if norm(center_delta) <= 1e-12:
                return 0.0
            separation_direction = normalized(center_delta)
            contact_delta = tuple(
                transforms[contact_a][index][3] - transforms[contact_b][index][3]
                for index in range(3)
            )
            return dot(contact_delta, separation_direction)
        matrix_a = self.geometry_transform(endpoint_a.geometry, transforms)
        matrix_b = self.geometry_transform(endpoint_b.geometry, transforms)
        center_a = transform_point(matrix_a, (0.0, 0.0, 0.0))
        center_b = transform_point(matrix_b, (0.0, 0.0, 0.0))
        delta = tuple(center_a[i] - center_b[i] for i in range(3))
        distance = norm(delta)
        if distance <= 1e-12:
            return 0.0
        direction = tuple(value / distance for value in delta)
        dims_a = self.geometry_dimensions(endpoint_a.geometry, mesh_intrinsic_dimensions)
        dims_b = self.geometry_dimensions(endpoint_b.geometry, mesh_intrinsic_dimensions)
        if dims_a is None or dims_b is None:
            return distance

        def support(matrix: Mat4, dims: Vec3) -> float:
            return sum(
                abs(dot(direction, transform_vector(matrix, axis))) * dims[index] / 2.0
                for index, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
            )

        return distance - support(matrix_a, dims_a) - support(matrix_b, dims_b)

    def inferred_open_closed(
        self,
        pair: GripperPair,
        endpoint_a: Endpoint,
        endpoint_b: Endpoint,
        mesh_intrinsic_dimensions: Callable[[Path], Vec3] | None = None,
    ) -> tuple[float, float]:
        master = self.master_joint(pair)
        lower, upper = self.joint_limits(master)
        lower_gap = self.opening_gap(pair, endpoint_a, endpoint_b, lower, mesh_intrinsic_dimensions)
        upper_gap = self.opening_gap(pair, endpoint_a, endpoint_b, upper, mesh_intrinsic_dimensions)
        if lower_gap is not None and upper_gap is not None and lower_gap > upper_gap:
            return upper, lower
        return lower, upper

    def resolve_mesh_filename(self, filename: str) -> Path | None:
        if not filename or filename.startswith("package://") or filename.startswith("model://"):
            return None
        if filename.startswith("file://"):
            candidate = Path(filename[7:])
        else:
            candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = self.source_path.parent / candidate
        return candidate.resolve()

    def _export_root(self, target: Path, *, rebase_mesh_paths: bool) -> ET.Element:
        root = copy.deepcopy(self.root)
        if rebase_mesh_paths:
            for mesh in root.findall(".//mesh"):
                filename = mesh.get("filename", "")
                resolved = self.resolve_mesh_filename(filename)
                if resolved is None or Path(filename).is_absolute() or filename.startswith("file://"):
                    continue
                relative = os.path.relpath(resolved, target.parent)
                mesh.set("filename", Path(relative).as_posix())
        return root

    def export(
        self,
        target_path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        rebase_mesh_paths: bool = True,
    ) -> Path:
        target = Path(target_path).expanduser().resolve()
        if target == self.source_path:
            raise URDFValidationError("Refusing to overwrite the input URDF; export to a new file")
        if target.exists() and not overwrite:
            raise URDFValidationError(f"Output already exists: {target}")
        if not target.parent.is_dir():
            raise URDFValidationError(f"Output directory does not exist: {target.parent}")
        self.validate()
        root = self._export_root(target, rebase_mesh_paths=rebase_mesh_paths)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            parsed = ET.parse(temporary, parser=parser)
            URDFDocument(target, parsed).validate()
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def iter_geometry(document: URDFDocument, links: Iterable[str] | None = None) -> Iterator[GeometryRef]:
    selected = set(links) if links is not None else set(document.links)
    for link_name in document.links:
        if link_name in selected:
            yield from document.geometry_refs(link_name)
