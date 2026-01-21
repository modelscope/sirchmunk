# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, List


def _download_and_extract(url: str, ext: str, required_bins: List[str], install_dir: Path, bin_label: str):
    """Downloads and extracts specific binaries from an archive."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = Path(tmp_file.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, tmp_file)

        temp_extract_dir = Path(tempfile.mkdtemp())
        if ext == ".zip":
            with zipfile.ZipFile(tmp_path, "r") as zf:
                for member in zf.namelist():
                    fname = os.path.basename(member)
                    if fname in required_bins:
                        with zf.open(member) as source, open(install_dir / fname, "wb") as f:
                            shutil.copyfileobj(source, f)
                        (install_dir / fname).chmod(0o755)
        else:  # .tar.gz
            with tarfile.open(tmp_path, "r:gz") as tf:
                for member in tf.getmembers():
                    fname = os.path.basename(member.name)
                    if fname in required_bins:
                        # Fix for Python 3.14 DeprecationWarning
                        tf.extract(member, temp_extract_dir, filter='data')
                        target = install_dir / fname
                        shutil.move(str(temp_extract_dir / member.name), str(target))
                        target.chmod(0o755)
    finally:
        if 'tmp_path' in locals(): tmp_path.unlink(missing_ok=True)
        if 'temp_extract_dir' in locals(): shutil.rmtree(temp_extract_dir, ignore_errors=True)


def _verify_bin(path: Path, expected_name: str) -> bool:
    """Check if binary exists and responds to --version."""
    if not path.exists(): return False
    try:
        res = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=5)
        return res.returncode == 0
    except:
        return False


def _install_component(repo: str, bin_name: str, required_bins: List[str], install_dir: Path, force: bool) -> str:
    """Generic installer for ripgrep and rga."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Platform mapping
    arch = "x86_64" if machine in ("x86_64", "amd64") else "aarch64" if machine in ("arm64", "aarch64") else None
    if not arch: raise RuntimeError(f"Unsupported arch: {machine}")

    if system == "windows":
        os_tag, ext = "pc-windows-msvc", ".zip"
    elif system == "darwin":
        os_tag, ext = "apple-darwin", ".tar.gz"
    else:
        # ripgrep and rga both use musl for static linux binaries
        os_tag, ext = "unknown-linux-musl", ".tar.gz"

    final_bin = install_dir / (bin_name + (".exe" if system == "windows" else ""))

    if not force and _verify_bin(final_bin, bin_name):
        return str(final_bin)

    print(f"Installing {bin_name} from {repo}...", file=sys.stderr)
    try:
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assets = json.loads(resp.read())["assets"]

        # Find asset (ripgrep assets often contain 'x86_64-unknown-linux-musl')
        asset = next(a for a in assets if arch in a["name"] and os_tag in a["name"] and a["name"].endswith(ext))
        _download_and_extract(asset["browser_download_url"], ext, required_bins, install_dir, bin_name)

        if not _verify_bin(final_bin, bin_name):
            raise RuntimeError(f"Verification failed for {bin_name}")
        return str(final_bin)
    except Exception as e:
        raise RuntimeError(f"Failed to install {bin_name}: {e}")


def install_rga(force_reinstall: bool = False, install_dir: Optional[str] = None) -> str:
    """Main entry: Installs ripgrep (rg) then ripgrep-all (rga)."""
    if install_dir is None:
        if platform.system().lower() == "windows":
            install_dir = os.path.expandvars(r"%LOCALAPPDATA%\bin")
        else:
            install_dir = os.path.expanduser("~/.local/bin")

    path_dir = Path(install_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install ripgrep (rg)
    rg_exe = "rg.exe" if platform.system().lower() == "windows" else "rg"
    _install_component("BurntSushi/ripgrep", "rg", [rg_exe], path_dir, force_reinstall)

    # 2. Install ripgrep-all (rga)
    rga_bins = ["rga.exe", "rga-preproc.exe"] if platform.system().lower() == "windows" else ["rga", "rga-preproc"]
    return _install_component("phiresky/ripgrep-all", "rga", rga_bins, path_dir, force_reinstall)


if __name__ == "__main__":
    try:
        path = install_rga()
        print(f"SUCCESS: ripgrep and ripgrep-all are ready at: {os.path.dirname(path)}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
