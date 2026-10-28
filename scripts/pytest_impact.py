#!/usr/bin/env python3
"""Conservative, explicit change selection for non-hardware acceptance.

This is not a general Python dependency oracle. Only reviewed independent
owners may omit seven costly B601 control modules. Everything else runs; an
unknown path or dependency restores the complete suite. No test result is
cached or synthesized for an omitted module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

SCHEMA = "agenticrobot.pytest-impact.v1"
B601_CONTROL_MODULES = frozenset(
    {
        "tests/test_b601_automatic_calibration.py",
        "tests/test_b601_calibration_cycle.py",
        "tests/test_b601_coordinated_holding.py",
        "tests/test_b601_cruise_velocity.py",
        "tests/test_b601_workspace_calibration.py",
        "tests/test_b601_workspace_identification.py",
        "tests/test_rebot_b601_staged_assist.py",
    }
)
INDEPENDENT_ROOTS = ("agent_gateway/", "deployments/perception/", "docs/agent_gateway/")
# These audited dynamic imports are package-local literal lazy-export tables,
# or the pinned Pin/Coal extension inspection in _sdk. Include every literal
# lazy target below. Any content drift requires a new review, otherwise full.
DYNAMIC_IMPORT_AUDITS = {
    "control_station/__init__.py": "7a3f64738e85f6572e861aef5d8cf82b6062addd7cb025581d989a4013c792ed",
    "robot_motion/__init__.py": "a048f446659faea95e4b4bda9d4485d9b3870852c1ec6f61056b2472759c3c71",
    "dataset_contracts/__init__.py": "f052ef8dc0b973d83ca08c5e02881eef1175f469430e56052c2da33a52d99f50",
    "dataset_interface/__init__.py": "d294081b15f356df26a1c3777038c40e598080d8fbf38842c616290ea6b15601",
    "robot_motion/rm65/arm_init/__init__.py": "5736d807dbaaebae99642837cb26b6dd6488f2ccf00ead31f709e254608d4290",
    "robot_motion/rm65/ik_checkpoint.py": "0e283c5c3aa0a2e4fd7dd9e2ec5cb8a843040ef58e59bfa5d0dadb241269418f",
}
DYNAMIC_CALLS = {
    "import_module",
    "__import__",
    "spec_from_file_location",
    "exec_module",
    "exec",
    "eval",
    "run_module",
    "run_path",
    "load_module",
    "find_spec",
    "entry_points",
    "iter_entry_points",
    "SourceFileLoader",
    "SourcelessFileLoader",
}


def unknown_loader_module(name: str) -> bool:
    return (
        name == "importlib"
        or name.startswith("importlib.")
        or name.split(".", 1)[0] in {"runpy", "pkgutil", "pkg_resources"}
    )


def git(repository: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, input=input, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def commit(repository: Path, ref: str) -> str:
    return (
        git(repository, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}")
        .decode()
        .strip()
    )


def clean(repository: Path) -> bool:
    return not git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )


class Snapshot:
    def __init__(
        self,
        repository: Path,
        revision: str,
        objects: dict | None = None,
        dependency_cache: dict | None = None,
    ):
        self.repository = repository
        self.commit = commit(repository, revision)
        self.files = {}
        for record in git(
            repository, "ls-tree", "-rz", "--full-tree", self.commit
        ).split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            path = raw_path.decode("utf-8")
            parts = PurePosixPath(path).parts
            if (
                not parts
                or path.startswith("/")
                or ".." in parts
                or "\n" in path
                or "\r" in path
            ):
                raise ValueError("unsupported Git path")
            mode, kind, oid = metadata.decode().split()
            self.files[path] = (mode, kind, oid)
        self.modules = {}
        for path, (mode, kind, _) in self.files.items():
            # Include symlink candidates in resolution so a protected import
            # through one fails closed in preload instead of disappearing.
            if path.endswith(".py") and kind == "blob":
                name = path[:-3].replace("/", ".").removesuffix(".__init__")
                self.modules.setdefault(name, set()).add(path)
                if path.startswith("tests/") and path.count("/") == 1:
                    # Include pytest's existing test-root imports. Keep all
                    # candidates when names collide rather than guess sys.path.
                    self.modules.setdefault(path[6:-3], set()).add(path)
        self._objects = objects if objects is not None else {}
        self.dependency_cache = dependency_cache if dependency_cache is not None else {}
        # Import and literal resource resolution depend on the tree's names
        # and kinds, as well as each source blob. Adding a previously missing
        # module must invalidate reuse of an unchanged consumer's analysis.
        self.layout = hashlib.sha256(
            json.dumps(
                [(path, mode, kind) for path, (mode, kind, _oid) in self.files.items()]
            ).encode()
        ).hexdigest()

    def preload(self, paths) -> None:
        missing = set()
        for path in paths:
            mode, kind, oid = self.files[path]
            if (mode, kind) == ("160000", "commit"):
                continue
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError(f"non-regular dependency: {path}")
            if oid not in self._objects:
                missing.add(oid)
        if not missing:
            return
        data = git(
            self.repository,
            "cat-file",
            "--batch",
            input=("\n".join(sorted(missing)) + "\n").encode(),
        )
        offset = 0
        for oid in sorted(missing):
            end = data.index(b"\n", offset)
            header = data[offset:end].decode().split()
            if len(header) != 3 or header[:2] != [oid, "blob"]:
                raise ValueError("unexpected Git object response")
            length = int(header[2])
            offset = end + 1
            self._objects[oid] = data[offset : offset + length]
            offset += length + 1
        if offset != len(data):
            raise ValueError("incomplete Git object response")

    def read(self, path: str) -> bytes:
        self.preload([path])
        return self._objects[self.files[path][2]]

    def resolve(self, module: str) -> set[str]:
        parts = module.split(".")
        return {
            path
            for n in range(1, len(parts) + 1)
            if (name := ".".join(parts[:n])) in self.modules
            for path in self.modules[name]
        }

    def absolute(self, path: str, module: str) -> str:
        if not module.startswith("."):
            return module
        package = path[:-3].split("/")[:-1]
        level = len(module) - len(module.lstrip("."))
        if level > len(package):
            raise ValueError(f"relative import escapes package: {path}")
        return ".".join(
            [*package[: len(package) - level + 1], module.lstrip(".")]
        ).rstrip(".")


def protected_dependencies(snapshot: Snapshot, roots: set[str]) -> set[str]:
    pending = list(roots)
    seen = set()
    while pending:
        snapshot.preload(p for p in pending if p not in seen)
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        # A gitlink is an independently owned, version-pinned source boundary.
        # Changes to any gitlink always select full acceptance. Checkout
        # cleanliness (including submodules) is checked separately by the runner.
        if snapshot.files[path][:2] == ("160000", "commit"):
            continue
        raw = snapshot.read(path)
        if not path.endswith(".py"):
            continue
        cache_key = (path, snapshot.files[path][2], snapshot.layout)
        if cache_key in snapshot.dependency_cache:
            pending.extend(snapshot.dependency_cache[cache_key] - seen)
            continue
        source = raw.decode("utf-8")
        # pytest_plugins may register fixtures/hooks globally without a Python
        # import edge. This first policy deliberately declines that mechanism.
        if re.search(r"\bpytest_plugins\b", source):
            raise ValueError(f"unresolved global pytest_plugins dependency: {path}")
        tree = ast.parse(source, filename=path)
        nodes = list(ast.walk(tree))
        dependencies = set()
        aliases = {name: name for name in DYNAMIC_CALLS}
        audited_dynamic = (
            DYNAMIC_IMPORT_AUDITS.get(path) == hashlib.sha256(raw).hexdigest()
        )
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if unknown_loader_module(alias.name) and not audited_dynamic:
                        raise ValueError(f"unaudited module loader: {path}")
                    dependencies.update(snapshot.resolve(alias.name))
            elif isinstance(node, ast.ImportFrom):
                name = snapshot.absolute(path, "." * node.level + (node.module or ""))
                supported_loader = (
                    name == "importlib"
                    and all(a.name == "import_module" for a in node.names)
                ) or (
                    name == "importlib.metadata"
                    and all(
                        a.name in {"version", "PackageNotFoundError"}
                        for a in node.names
                    )
                )
                if (
                    unknown_loader_module(name)
                    and not supported_loader
                    and not audited_dynamic
                ):
                    raise ValueError(f"unaudited module loader: {path}")
                dependencies.update(snapshot.resolve(name))
                for alias in node.names:
                    if alias.name in DYNAMIC_CALLS:
                        aliases[alias.asname or alias.name] = alias.name
                    if alias.name != "*":
                        dependencies.update(snapshot.resolve(name + "." + alias.name))
        # Package roots expose literal, lazy public APIs. Traverse all targets,
        # including unused exports; this intentionally overestimates impact.
        for node in tree.body:
            names = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(
                isinstance(n, ast.Name) and n.id in {"_LAZY_EXPORTS", "_EXPORTS"}
                for n in names
            ):
                mapping = ast.literal_eval(node.value)
                if not isinstance(mapping, dict):
                    raise ValueError(f"unsupported lazy export table: {path}")
                for value in mapping.values():
                    module = (
                        value
                        if isinstance(value, str)
                        else value[0]
                        if isinstance(value, (tuple, list)) and value
                        else None
                    )
                    if not isinstance(module, str):
                        raise TypeError(f"unsupported lazy export target: {path}")
                    dependencies.update(
                        snapshot.resolve(snapshot.absolute(path, module))
                    )
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            function = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            function = aliases.get(function, function)
            if function not in DYNAMIC_CALLS:
                continue
            if (
                function in {"import_module", "__import__"}
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and not node.args[0].value.startswith(".")
            ):
                dependencies.update(snapshot.resolve(node.args[0].value))
            elif not audited_dynamic:
                raise ValueError(f"unaudited dynamic dependency: {path}:{node.lineno}")
        loader_names = [
            node
            for node in nodes
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in aliases
        ]
        if loader_names and not audited_dynamic:
            parents = {
                id(child): node
                for node in nodes
                for child in ast.iter_child_nodes(node)
            }
            for node in loader_names:
                parent = parents.get(id(node))
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    raise ValueError(f"escaped module loader: {path}:{node.lineno}")
        # Resource paths or Python snippets can bypass ordinary import edges.
        # A possible reference to the independent owners makes them protected.
        for node in nodes:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip().removeprefix("./")
                if value in snapshot.files:
                    dependencies.add(value)
                for prefix in (
                    "agent_gateway",
                    "deployments/perception",
                    "docs/agent_gateway",
                ):
                    if (
                        value == prefix
                        or value.startswith(
                            (prefix + "/", prefix.replace("/", ".") + ".")
                        )
                        or value in prefix.split("/")
                        or value == prefix.replace("/", ".")
                        or re.search(
                            r"\b(?:from|import)\s+"
                            + re.escape(prefix.replace("/", ".")),
                            value,
                        )
                    ):
                        dependencies.update(
                            p for p in snapshot.files if p.startswith(prefix + "/")
                        )
        snapshot.dependency_cache[cache_key] = dependencies
        pending.extend(dependencies - seen)
    return seen


def independent_path(path: str) -> bool:
    owned = (
        path.startswith(INDEPENDENT_ROOTS)
        or (
            path.startswith("tests/test_agent_gateway_")
            and path.endswith(".py")
            and PurePosixPath(path).parent == PurePosixPath("tests")
        )
        or path == "tests/test_sam3_service.py"
    )
    return (
        owned
        and PurePosixPath(path).suffix in {".py", ".json", ".md"}
        and PurePosixPath(path).name != "__init__.py"
    )


def selection(repository: Path, base: str, head: str = "HEAD") -> dict:
    repository = repository.resolve()
    objects, dependency_cache = {}, {}
    before = Snapshot(repository, base, objects, dependency_cache)
    after = Snapshot(repository, head, objects, dependency_cache)
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", before.commit, after.commit],
        cwd=repository,
        check=False,
    )
    if ancestry.returncode:
        raise ValueError("change base must be an ancestor of the selected head")
    all_modules = sorted(
        p
        for p in after.files
        if p.startswith("tests/")
        and PurePosixPath(p).name.startswith("test_")
        and p.endswith(".py")
    )
    if not all_modules:
        raise ValueError("no tracked pytest modules")
    changed = sorted(
        p
        for p in before.files.keys() | after.files.keys()
        if before.files.get(p) != after.files.get(p)
    )
    reasons = []
    if not changed:
        reasons.append("no changed paths; no prior test result is reused")
    for path in changed:
        old, new = before.files.get(path), after.files.get(path)
        if not independent_path(path):
            reasons.append(f"shared or unclassified path: {path}")
        elif (
            new is None
            or new[0] not in {"100644", "100755"}
            or (old and old[0] != new[0])
        ):
            reasons.append(f"deletion or file-mode change: {path}")
        else:
            try:
                text = after.read(path).decode("utf-8")
            except UnicodeError:
                reasons.append(f"unreadable changed owner: {path}")
                continue
            if re.search(r"b601|rebot_b601", text, re.IGNORECASE):
                reasons.append(f"explicit B601 reference in changed owner: {path}")
            if path.endswith(".py") and re.search(r"\bpytest_plugins\b", text):
                reasons.append(
                    f"global pytest_plugins declaration in changed owner: {path}"
                )
    excluded = sorted(B601_CONTROL_MODULES & set(all_modules))
    protected = set()
    if not reasons and excluded:
        try:
            for snapshot in (before, after):
                roots = set(excluded) | {"tests/conftest.py"}
                if not roots <= snapshot.files.keys():
                    raise ValueError(
                        "protected test or shared fixture missing from a snapshot"
                    )
                protected.update(protected_dependencies(snapshot, roots))
            for path in sorted(set(changed) & protected):
                reasons.append(f"B601 import/resource dependency: {path}")
        except (ValueError, SyntaxError, UnicodeError, KeyError, TypeError) as error:
            reasons.append(f"dependency analysis cannot establish isolation: {error}")
    elif not excluded:
        reasons.append("no reviewed B601 control modules present")
    if reasons:
        excluded = []
    selected = sorted(set(all_modules) - set(excluded))
    return {
        "schema": SCHEMA,
        "mode": "affected" if excluded else "full",
        "base_commit": before.commit,
        "head_commit": after.commit,
        "head_tree": git(repository, "rev-parse", after.commit + "^{tree}")
        .decode()
        .strip(),
        "changed_paths": changed,
        "fallback_reasons": reasons,
        "excluded_modules": excluded,
        "selected_modules": selected,
        "protected_dependencies": sorted(protected),
        "scope": "All non-hardware tests except explicitly listed unchanged B601 control modules; not complete-suite evidence when mode=affected.",
    }


def verify_checkout(repository: Path, report: dict) -> None:
    if not clean(repository) or commit(repository, "HEAD") != report["head_commit"]:
        raise ValueError(
            "acceptance source changed or is dirty; selection evidence is invalid"
        )
    discovered = {
        p.relative_to(repository).as_posix()
        for p in (repository / "tests").rglob("test_*.py")
        if p.is_file()
    }
    tracked = set(report["selected_modules"]) | set(report["excluded_modules"])
    if discovered != tracked:
        raise ValueError("filesystem test discovery differs from the pinned Git tree")
    if any((repository / p).is_symlink() for p in tracked):
        raise ValueError("change-aware acceptance requires regular test modules")


def selection_for_shards(repository: Path, base: str, shard_count: int) -> dict:
    from pytest_sharding import plan_shards

    report = selection(repository, base)
    report["shard_count"] = shard_count
    report["shard_layout"] = "filtered-complete-plan.v1"
    shards = plan_shards(
        repository,
        repository / "tests",
        shard_count,
        "full",
        test_modules=report["selected_modules"],
    )
    if report["mode"] == "affected" and any(not shard for shard in shards):
        report["mode"] = "full"
        report["selected_modules"] = sorted(
            set(report["selected_modules"]) | set(report["excluded_modules"])
        )
        report["excluded_modules"] = []
        report["fallback_reasons"].append(
            "filtering the complete plan would empty a configured shard; retain full acceptance"
        )
    return report


def verify(repository: Path, path: Path, shard_count: int) -> None:
    repository = repository.resolve()
    report = json.loads(path.read_text())
    verify_checkout(repository, report)
    expected = selection_for_shards(repository, report["base_commit"], shard_count)
    if {key: report.get(key) for key in expected} != expected:
        raise ValueError("selection does not match the pinned Git inputs and policy")
    if hashlib.sha256(
        path.with_name("shard-plan.json").read_bytes()
    ).hexdigest() != report.get("shard_plan_sha256"):
        raise ValueError("shard plan changed after impact selection")
    for name, digest in report["paths_sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256(path.with_name(name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("shard targets changed after impact selection")
    # Hashes detect ordinary corruption; they are not proof of coverage.
    # Independently bind the actual path lists and the complete target set to
    # the source and the runner's configured shard count, even if report and
    # plan have been changed together into a self-consistent incomplete set.
    from pytest_sharding import plan_shards

    plan = json.loads(path.with_name("shard-plan.json").read_text())
    if (
        plan["schema"] != "dexteleop.pytest-shard-plan.v2"
        or plan["profile"] != "full"
        or plan["shard_count"] != shard_count
        or len(plan["shards"]) != shard_count
    ):
        raise ValueError("shard plan profile/count differs from the running wrapper")
    target_lists = [shard["test_targets"] for shard in plan["shards"]]
    targets = [target for items in target_lists for target in items]
    if len(targets) != len(set(targets)) or {
        target.split("::", 1)[0] for target in targets
    } != set(report["selected_modules"]):
        raise ValueError("shard target coverage differs from the selected test modules")
    if plan["file_count"] != len(report["selected_modules"]) or plan[
        "target_count"
    ] != len(targets):
        raise ValueError("shard plan counts differ from the selected target coverage")
    shards = plan_shards(
        repository,
        repository / "tests",
        shard_count,
        "full",
        test_modules=report["selected_modules"],
    )
    expected_paths = {f"shard-{i}-paths.txt" for i in range(shard_count)}
    if set(report["paths_sha256"]) != expected_paths:
        raise ValueError(
            "shard target path list coverage differs from the running wrapper"
        )
    for index, shard in enumerate(shards):
        planned = plan["shards"][index]
        if (
            planned["index"] != index
            or planned["paths_file"] != f"shard-{index}-paths.txt"
            or planned["test_targets"] != [item[0] for item in shard]
        ):
            raise ValueError(
                "shard target plan differs from the complete source-derived plan"
            )
        modules = sorted({item[0].split("::", 1)[0] for item in shard})
        if (
            planned["test_modules"] != modules
            or planned["file_count"] != len(modules)
            or planned["target_count"] != len(shard)
        ):
            raise ValueError(
                "shard module counts differ from the complete source-derived plan"
            )
        content = "".join(f"{item[1]}\n" for item in shard).encode()
        if path.with_name(planned["paths_file"]).read_bytes() != content:
            raise ValueError(
                "shard targets differ from the complete source-derived path list"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "plan", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--repository", type=Path, required=True)
        if name == "verify":
            sub.add_argument("--selection", type=Path, required=True)
            sub.add_argument("--junit", type=Path)
            sub.add_argument("--shards", type=int, required=True)
        else:
            sub.add_argument("--base", required=True)
            if name == "inspect":
                sub.add_argument("--head", default="HEAD")
            else:
                sub.add_argument("--output-dir", type=Path, required=True)
                sub.add_argument("--shards", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            verify(args.repository, args.selection, args.shards)
            if args.junit:
                report = json.loads(args.selection.read_text())
                document = ET.parse(args.junit)
                document.getroot().set("name", "changed-acceptance")
                for suite in document.getroot().iter("testsuite"):
                    properties = suite.find("properties")
                    if properties is None:
                        properties = ET.Element("properties")
                        suite.insert(0, properties)
                    for key, value in {
                        "acceptance.kind": "change-aware",
                        "acceptance.mode": report["mode"],
                        "acceptance.base": report["base_commit"],
                        "acceptance.head": report["head_commit"],
                        "acceptance.excluded_modules": json.dumps(
                            report["excluded_modules"]
                        ),
                    }.items():
                        ET.SubElement(properties, "property", name=key, value=value)
                document.write(args.junit, encoding="utf-8", xml_declaration=True)
            return 0
        if args.command == "inspect":
            report = selection(args.repository, args.base, args.head)
            report["advisory_only"] = True
            report["shard_configuration_checked"] = False
            report["checkout_clean"] = clean(args.repository)
            print(json.dumps(report, indent=2))
            return 0
        report = selection_for_shards(args.repository.resolve(), args.base, args.shards)
        verify_checkout(args.repository.resolve(), report)
        from pytest_sharding import write_plan

        write_plan(
            args.repository,
            args.repository / "tests",
            args.output_dir,
            args.shards,
            "full",
            "shard",
            test_modules=report["selected_modules"],
        )
        report["shard_plan_sha256"] = hashlib.sha256(
            (args.output_dir / "shard-plan.json").read_bytes()
        ).hexdigest()
        report["paths_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output_dir.glob("shard-*-paths.txt"))
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "mode",
                        "base_commit",
                        "head_commit",
                        "excluded_modules",
                        "fallback_reasons",
                    )
                }
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        SyntaxError,
        UnicodeError,
        KeyError,
        TypeError,
        ET.ParseError,
    ) as error:
        print(f"pytest_impact: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
