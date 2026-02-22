#!/usr/bin/env python3
"""
Build script for Sirchmunk Docker images.

Design follows the modelscope/modelscope docker/build_image.py pattern:
  - A ``Builder`` base class handles Dockerfile template rendering, build and push.
  - ``CPUImageBuilder`` (default) produces a lightweight CPU-only image.
  - Multi-registry push to Alibaba Cloud ACR (cn-beijing, cn-hangzhou, us-west-1).

Usage:
    # Local build (no push)
    python docker/build_image.py

    # Dry-run (generate Dockerfile only)
    python docker/build_image.py --dry_run 1

    # Build with Chinese mirror acceleration (for China mainland networks)
    python docker/build_image.py --mirror cn

    # Build and push to default ACR registries
    python docker/build_image.py --push

    # Build and push to custom registries
    python docker/build_image.py --push --registries "my-registry.example.com/ns/sirchmunk"
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Default ACR registries (Alibaba Cloud Container Registry)
# ---------------------------------------------------------------------------

DEFAULT_ACR_REGISTRIES = [
    "modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk",
    "modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk",
    "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk",
]

# ---------------------------------------------------------------------------
# Mirror configuration for China mainland
# ---------------------------------------------------------------------------

MIRROR_PROFILES: Dict[str, Dict[str, str]] = {
    "cn": {
        "docker_prefix": "docker.m.daocloud.io/library/",
        "pip_index_url": "https://mirrors.aliyun.com/pypi/simple/",
        "pip_trusted_host": "mirrors.aliyun.com",
        "npm_registry": "https://registry.npmmirror.com",
        "github_proxy": "https://ghfast.top/",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_version_from_source() -> str:
    """Read ``__version__`` from ``src/sirchmunk/version.py``."""
    version_file = REPO_ROOT / "src" / "sirchmunk" / "version.py"
    if not version_file.exists():
        return "latest"
    text = version_file.read_text()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else "latest"


def _generate_python_tag(python_version: str) -> str:
    """``'3.12'`` → ``'py312'``, ``'3.12.1'`` → ``'py312'``."""
    parts = python_version.split(".")[:2]
    return "py" + "".join(parts)


# ---------------------------------------------------------------------------
# Base Builder
# ---------------------------------------------------------------------------

class Builder:
    """Abstract builder that renders a Dockerfile template, builds and pushes."""

    DEFAULTS = {
        "python_version": "3.12",
        "node_version": "20",
        "ubuntu_version": "22.04",
        "rg_version": "14.1.1",
        "rga_version": "v1.0.0-alpha.5",
        "port": "8584",
    }

    def __init__(self, args: Any, dry_run: bool = False):
        self.args = self._init_args(args)
        self.dry_run = dry_run
        self.mirror: Optional[Dict[str, str]] = MIRROR_PROFILES.get(
            getattr(args, "mirror", None) or ""
        )
        self.registries: List[str] = self._resolve_registries()

    def _init_args(self, args: Any) -> Any:
        for key, default in self.DEFAULTS.items():
            if not getattr(args, key, None):
                setattr(args, key, default)
        if not getattr(args, "sirchmunk_version", None):
            args.sirchmunk_version = _read_version_from_source()
        return args

    def _resolve_registries(self) -> List[str]:
        """Parse --registries or fall back to default ACR list when --push."""
        raw = getattr(self.args, "registries", None)
        if raw:
            return [r.strip() for r in raw.split(",") if r.strip()]
        if getattr(self.args, "push", False):
            return list(DEFAULT_ACR_REGISTRIES)
        return []

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _template_path(self) -> Path:
        return REPO_ROOT / "docker" / "Dockerfile.ubuntu"

    def _mirror_replacements(self) -> dict:
        if self.mirror:
            pip_args = (
                f"-i {self.mirror['pip_index_url']} "
                f"--trusted-host {self.mirror['pip_trusted_host']} "
            )
            npm_cmd = f"RUN npm config set registry {self.mirror['npm_registry']}\n"
            github_proxy = self.mirror["github_proxy"]
        else:
            pip_args = ""
            npm_cmd = ""
            github_proxy = ""
        return {
            "pip_index_args": pip_args,
            "npm_mirror_cmd": npm_cmd,
            "github_proxy": github_proxy,
        }

    def _replacements(self) -> dict:
        raise NotImplementedError

    def generate_dockerfile(self) -> str:
        content = self._template_path().read_text()
        replacements = {**self._replacements(), **self._mirror_replacements()}
        for key, value in replacements.items():
            content = content.replace(f"{{{key}}}", value)
        return content

    # ------------------------------------------------------------------
    # Image tag — follows ModelScope convention
    #   ubuntu22.04-py312-0.0.2
    # ------------------------------------------------------------------

    def image_tag(self) -> str:
        raise NotImplementedError

    def local_image(self) -> str:
        """Local image name used during docker build."""
        return f"sirchmunk:{self.image_tag()}"

    # ------------------------------------------------------------------
    # Build & push
    # ------------------------------------------------------------------

    def _save_dockerfile(self, content: str) -> None:
        dest = REPO_ROOT / "Dockerfile"
        if dest.exists():
            dest.unlink()
        dest.write_text(content)
        print(f"[build_image] Generated {dest}")

    def build(self) -> int:
        ret = os.system(
            f"docker build --platform linux/amd64 -t {self.local_image()} -f Dockerfile ."
        )
        if ret == 0:
            os.system(f"docker tag {self.local_image()} sirchmunk:latest")
        return ret

    def push(self) -> int:
        tag = self.image_tag()
        for registry in self.registries:
            remote = f"{registry}:{tag}"
            print(f"[build_image] Pushing → {remote}")

            ret = os.system(f"docker tag {self.local_image()} {remote}")
            if ret != 0:
                return ret
            ret = os.system(f"docker push {remote}")
            if ret != 0:
                return ret

            ts_remote = f"{registry}:{tag}-{TIMESTAMP}"
            os.system(f"docker tag {self.local_image()} {ts_remote}")
            os.system(f"docker push {ts_remote}")

        return 0

    # ------------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------------

    def __call__(self) -> None:
        if self.mirror:
            print(f"[build_image] Using mirror profile: {self.args.mirror}")

        content = self.generate_dockerfile()
        self._save_dockerfile(content)

        if self.dry_run:
            print(f"[build_image] Dry-run complete.")
            print(f"[build_image] Local image: {self.local_image()}")
            if self.registries:
                for r in self.registries:
                    print(f"[build_image] Push target: {r}:{self.image_tag()}")
            return

        os.chdir(REPO_ROOT)

        ret = self.build()
        if ret != 0:
            raise RuntimeError(f"Docker build failed with exit code {ret}")
        print(f"[build_image] Built: {self.local_image()}")

        if self.registries:
            ret = self.push()
            if ret != 0:
                raise RuntimeError(f"Docker push failed with exit code {ret}")

        print(f"[build_image] Done: {self.local_image()}")


# ---------------------------------------------------------------------------
# CPU Image Builder (default)
# ---------------------------------------------------------------------------

class CPUImageBuilder(Builder):

    def _replacements(self) -> dict:
        a = self.args
        prefix = self.mirror["docker_prefix"] if self.mirror else ""
        return {
            "node_image": f"{prefix}node:{a.node_version}-slim",
            "python_image": f"{prefix}python:{a.python_version}-slim",
            "rg_version": a.rg_version,
            "rga_version": a.rga_version,
            "port": a.port,
            "image_tag": self.image_tag(),
        }

    def image_tag(self) -> str:
        a = self.args
        py_tag = _generate_python_tag(a.python_version)
        return f"ubuntu{a.ubuntu_version}-{py_tag}-{a.sirchmunk_version}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Sirchmunk Docker images")
    p.add_argument("--image_type", default="cpu", choices=["cpu"],
                    help="Image type to build (default: cpu)")
    p.add_argument("--python_version", default=None,
                    help="Python base image version (default: 3.12)")
    p.add_argument("--node_version", default=None,
                    help="Node.js base image version (default: 20)")
    p.add_argument("--ubuntu_version", default=None,
                    help="Ubuntu version label for image tag (default: 22.04)")
    p.add_argument("--rg_version", default=None,
                    help="ripgrep version (default: 14.1.1)")
    p.add_argument("--rga_version", default=None,
                    help="ripgrep-all version (default: v1.0.0-alpha.5)")
    p.add_argument("--port", default=None,
                    help="Exposed port (default: 8584)")
    p.add_argument("--sirchmunk_version", default=None,
                    help="Version label for image tag (default: auto from version.py)")
    p.add_argument("--sirchmunk_branch", default="main",
                    help="Git branch being built (for CI traceability)")
    p.add_argument("--mirror", default=None, choices=list(MIRROR_PROFILES.keys()),
                    help="Use mirror sources for China mainland (cn)")
    p.add_argument("--push", action="store_true",
                    help="Push to registries after build")
    p.add_argument("--registries", default=None,
                    help="Comma-separated registries to push to (overrides defaults)")
    p.add_argument("--dry_run", type=int, default=0,
                    help="1 = generate Dockerfile only, skip docker build")
    return p.parse_args()


BUILDERS = {
    "cpu": CPUImageBuilder,
}


def main() -> None:
    args = parse_args()

    builder_cls = BUILDERS.get(args.image_type.lower())
    if builder_cls is None:
        print(f"Unsupported image_type: {args.image_type}", file=sys.stderr)
        sys.exit(1)

    builder = builder_cls(args, dry_run=bool(args.dry_run))
    builder()


if __name__ == "__main__":
    main()
