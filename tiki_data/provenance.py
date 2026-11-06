"""Content fingerprints, including dirty source and declared model/data inputs."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess

from .artifacts import safe_path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with safe_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_sha256(path):
    """Canonicalize Git's LF/CRLF conversion for source, never for data/weights."""
    content = safe_path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def file_record(path):
    path = safe_path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True,
                            check=True, timeout=15)
    return result.stdout.decode("utf-8").strip()


def source_snapshot():
    # Git includes uncommitted and new source files, and excludes ignored generated artifacts.
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    paths = sorted({name for name in names if name and (Path(name).suffix == ".py"
                    or ("/" not in name and (name.startswith("requirements") or name == "pyproject.toml")))})
    files = {name: source_sha256(ROOT / name) if (ROOT / name).is_file() else "deleted" for name in paths}
    return {"sha256": digest(files), "files": files, "revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain")),
            "normalization": "CRLF_to_LF_for_source_only"}


def environment():
    packages = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()
                if dist.metadata["Name"]}
    result = {"python": platform.python_version(), "implementation": platform.python_implementation(),
              "platform": platform.platform(), "packages": dict(sorted(packages.items()))}
    result["sha256"] = digest(result)
    return result
