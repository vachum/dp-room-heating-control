import json
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PACKAGES = ("numpy", "pandas", "torch", "cvxpy", "scipy")


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
    except Exception:
        return "unavailable"

    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


def get_git_info() -> dict[str, object]:
    status_output = _git(["status", "--porcelain"])
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": status_output not in {"", "unavailable"},
    }


def get_package_versions(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        return {pkg: "unknown" for pkg in packages}

    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def build_provenance_payload(extra: dict | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "package_versions": get_package_versions(),
    }
    payload.update(get_git_info())
    if extra:
        payload.update(extra)
    return payload


def write_json(path: str | Path, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def stable_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_artifact_root(
    artifact_root: str | Path,
    experiment_tag: str | None = None,
) -> Path:
    root = Path(artifact_root)
    if experiment_tag:
        root = root / experiment_tag
    root.mkdir(parents=True, exist_ok=True)
    return root
