import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_v16.freeze import _environment, verify_research as verify_parent

from .config import V20Settings


DIRECTORY = Path("artifacts/research_v20")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest().upper()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def settings_dict():
    return json.loads(json.dumps(asdict(V20Settings()), default=str))


def freeze(root="."):
    root = Path(root)
    folder = root / DIRECTORY
    if any((folder / name).exists() for name in ("plan.lock.json", "run.started.json", "report.json")):
        raise RuntimeError("V20 already frozen or started; use a new revision")
    parent = verify_parent(root)
    # Snapshot inherited frozen inputs and every old research lock/report/code.
    protected = set(parent["data_sha256"]) | set(parent["code_sha256"]) | set(parent["dependency_sha256"])
    for version in range(3, 20):
        protected.update(p.relative_to(root).as_posix() for p in (root / f"research_v{version}").glob("*.py"))
        for name in ("plan.lock.json", "plan.lock.sha256", "protocol.json", "report.json", "run.started.json"):
            path = root / f"artifacts/research_v{version}" / name
            if path.exists():
                protected.add(path.relative_to(root).as_posix())
    code = sorted(p.relative_to(root).as_posix() for p in (root / "research_v20").glob("*.py"))
    code += ["tests/test_research_v20.py", "artifacts/research_v20/protocol.json"]
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_lock_sha256": parent["lock_sha256"], "environment": _environment(),
        "settings": settings_dict(),
        "protected_sha256": {name: digest(root / name) for name in sorted(protected)},
        "code_sha256": {name: digest(root / name) for name in code},
        "execution_authorized": False,
    }
    write_new(folder / "plan.lock.json", lock)
    with (folder / "plan.lock.sha256").open("x", encoding="utf-8") as handle:
        handle.write(digest(folder / "plan.lock.json") + "\n")
    return verify(root)


def verify(root="."):
    root = Path(root)
    folder = root / DIRECTORY
    actual = digest(folder / "plan.lock.json")
    if actual != (folder / "plan.lock.sha256").read_text().strip():
        raise RuntimeError("V20 lock digest mismatch")
    lock = json.loads((folder / "plan.lock.json").read_text(encoding="utf-8"))
    for name, expected in {**lock["protected_sha256"], **lock["code_sha256"]}.items():
        if not (root / name).is_file() or digest(root / name) != expected:
            raise RuntimeError(f"frozen file changed: {name}")
    if settings_dict() != lock["settings"] or _environment() != lock["environment"]:
        raise RuntimeError("frozen settings/environment changed")
    parent = verify_parent(root)
    if parent["lock_sha256"] != lock["parent_lock_sha256"]:
        raise RuntimeError("V16 parent lock changed")
    return {**lock, "lock_sha256": actual, "frozen_inputs_intact": True}
