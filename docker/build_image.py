#!/usr/bin/env python3
"""
Build script for Sirchmunk Docker images.

Design follows the modelscope/modelscope docker/build_image.py pattern:
  - A ``Builder`` base class handles Dockerfile template rendering, build and push.
  - ``CPUImageBuilder`` (default) produces a lightweight CPU-only image.

Usage:
    # Dry-run (generate Dockerfile only, no docker build)
    python docker/build_image.py --dry_run 1

    # Build locally
    python docker/build_image.py --image_type cpu

    # Build and push to a registry
    DOCKER_REGISTRY=ghcr.io/modelscope/sirchmunk \
        python docker/build_image.py --image_type cpu
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DOCKER_REGISTRY = os.environ.get("DOCKER_REGISTRY", "sirchmunk")
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

# Resolve repository root (one level up from this script)
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Base Builder
# ---------------------------------------------------------------------------

class Builder:
    """Abstract builder that renders a Dockerfile template, builds and pushes."""

    # Default versions — subclasses may override via ``init_args``.
    DEFAULTS = {
        "python_version": "3.13",
        "node_version": "20",
        "rg_version": "14.1.1",
        "rga_version": "v1.0.0-alpha.5",
        "port": "8584",
    }

    def __init__(self, args: Any, dry_run: bool = False):
        self.args = self._init_args(args)
        self.dry_run = dry_run

    # ------------------------------------------------------------------

    def _init_args(self, args: Any) -> Any:
        """Apply default values for any args not explicitly provided."""
        for key, default in self.DEFAULTS.items():
            if not getattr(args, key, None):
                setattr(args, key, default)
        return args

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _template_path(self) -> Path:
        return REPO_ROOT / "docker" / "Dockerfile.ubuntu"

    def _replacements(self) -> dict:
        """Return placeholder → value mapping for the Dockerfile template."""
        raise NotImplementedError

    def generate_dockerfile(self) -> str:
        """Read the template and substitute all ``{placeholder}`` tokens."""
        content = self._template_path().read_text()
        for key, value in self._replacements().items():
            content = content.replace(f"{{{key}}}", value)
        return content

    # ------------------------------------------------------------------
    # Image tag
    # ------------------------------------------------------------------

    def image_tag(self) -> str:
        raise NotImplementedError

    def image(self) -> str:
        return f"{DOCKER_REGISTRY}:{self.image_tag()}"

    # ------------------------------------------------------------------
    # Build & push
    # ------------------------------------------------------------------

    def _save_dockerfile(self, content: str) -> None:
        """Write the rendered Dockerfile to the repo root."""
        dest = REPO_ROOT / "Dockerfile"
        if dest.exists():
            dest.unlink()
        dest.write_text(content)
        print(f"[build_image] Generated {dest}")

    def build(self) -> int:
        return os.system(
            f"docker build -t {self.image()} -f Dockerfile ."
        )

    def push(self) -> int:
        return os.system(f"docker push {self.image()}")

    # ------------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------------

    def __call__(self) -> None:
        content = self.generate_dockerfile()
        self._save_dockerfile(content)

        if self.dry_run:
            print(f"[build_image] Dry-run complete. Image would be: {self.image()}")
            return

        # cd to repo root so COPY paths work
        os.chdir(REPO_ROOT)

        ret = self.build()
        if ret != 0:
            raise RuntimeError(f"Docker build failed with exit code {ret}")

        # Only push when a real registry is set
        if DOCKER_REGISTRY != "sirchmunk":
            ret = self.push()
            if ret != 0:
                raise RuntimeError(f"Docker push failed with exit code {ret}")

            # Tag with timestamp for traceability
            ts_image = f"{DOCKER_REGISTRY}:{self.image_tag()}-{TIMESTAMP}"
            os.system(f"docker tag {self.image()} {ts_image}")
            os.system(f"docker push {ts_image}")

        print(f"[build_image] Done: {self.image()}")


# ---------------------------------------------------------------------------
# CPU Image Builder (default)
# ---------------------------------------------------------------------------

class CPUImageBuilder(Builder):
    """Produces a CPU-only Sirchmunk image (Python + Node frontend)."""

    def _replacements(self) -> dict:
        a = self.args
        return {
            "node_image": f"node:{a.node_version}-slim",
            "python_image": f"python:{a.python_version}-slim",
            "rg_version": a.rg_version,
            "rga_version": a.rga_version,
            "port": a.port,
            "image_tag": self.image_tag(),
        }

    def image_tag(self) -> str:
        ver = getattr(self.args, "sirchmunk_version", "latest")
        return f"ubuntu-py{self.args.python_version}-{ver}-cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Sirchmunk Docker images")
    p.add_argument("--image_type", default="cpu", choices=["cpu"],
                    help="Image type to build (default: cpu)")
    p.add_argument("--python_version", default=None,
                    help="Python base image version (default: 3.13)")
    p.add_argument("--node_version", default=None,
                    help="Node.js base image version (default: 20)")
    p.add_argument("--rg_version", default=None,
                    help="ripgrep version (default: 14.1.1)")
    p.add_argument("--rga_version", default=None,
                    help="ripgrep-all version (default: v1.0.0-alpha.5)")
    p.add_argument("--port", default=None,
                    help="Exposed port (default: 8584)")
    p.add_argument("--sirchmunk_version", default="latest",
                    help="Version label for the image tag")
    p.add_argument("--sirchmunk_branch", default="main",
                    help="Git branch being built (for CI traceability)")
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
