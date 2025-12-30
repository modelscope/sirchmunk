import hashlib
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
from typing import Optional


def install_ripgrep(
    force_reinstall: bool = False,
    install_dir: Optional[str] = None,
    bin_name: str = "rg",
) -> str:
    """Automatically detect, download, and install ripgrep (rg) cross-platform.

    Installs to user-local bin directory (no sudo required).

    Args:
        force_reinstall: If True, reinstall even if rg is found.
        install_dir: Custom install directory (e.g., "/opt/tools").
                     Defaults to platform-appropriate user bin dir.
        bin_name: Desired binary name (default "rg"). On Windows, ".exe" is auto-appended.

    Returns:
        Path to installed rg binary (e.g., "/home/user/.local/bin/rg").

    Raises:
        RuntimeError: If download, extraction, or verification fails.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    elif machine == "armv7l":
        arch = "arm"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    # Determine OS tag for ripgrep release
    if system == "windows":
        os_tag = "pc-windows-msvc"
        ext = ".zip"
        bin_file = "rg.exe"
    elif system == "darwin":
        os_tag = "apple-darwin"
        ext = ".tar.gz"
        bin_file = "rg"
    elif system == "linux":
        # Detect musl (Alpine) vs glibc
        try:
            libc = subprocess.check_output(
                ["ldd", "--version"], stderr=subprocess.STDOUT, text=True
            )
            if "musl" in libc.lower():
                os_tag = "unknown-linux-musl"
            else:
                os_tag = "unknown-linux-gnu"
        except Exception:
            # Fallback to glibc
            os_tag = "unknown-linux-gnu"
        ext = ".tar.gz"
        bin_file = "rg"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # Set installation directory
    if install_dir is None:
        if system == "windows":
            install_dir = os.path.expandvars(r"%LOCALAPPDATA%\bin")
        else:
            install_dir = os.path.expanduser("~/.local/bin")
            # Fallback: ~/bin if ~/.local/bin doesn't exist and ~/bin does
            if not os.path.exists(install_dir) and os.path.exists(
                os.path.expanduser("~/bin")
            ):
                install_dir = os.path.expanduser("~/bin")
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    # Final binary path
    final_bin = install_dir / (bin_name + (".exe" if system == "windows" else ""))

    # Check if already installed (and skip unless forced)
    if not force_reinstall:
        # Try current PATH first
        rg_in_path = shutil.which("rg")
        if rg_in_path:
            try:
                ver = subprocess.run(
                    [rg_in_path, "--version"], capture_output=True, text=True
                )
                if ver.returncode == 0 and "ripgrep" in ver.stdout:
                    return rg_in_path
            except Exception:
                pass

        # Check install_dir
        if final_bin.exists():
            try:
                ver = subprocess.run(
                    [str(final_bin), "--version"], capture_output=True, text=True
                )
                if ver.returncode == 0 and "ripgrep" in ver.stdout:
                    return str(final_bin)
            except Exception:
                pass

    # === Download latest release ===
    print("[ripgrep] Detecting latest version...", file=sys.stderr)
    try:
        # Get latest release tag
        with urllib.request.urlopen(
            "https://api.github.com/repos/BurntSushi/ripgrep/releases/latest",
            timeout=10,
        ) as resp:
            release_info = json.loads(resp.read())
        version = release_info["tag_name"].lstrip("v")
        assets = release_info["assets"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ripgrep release info: {e}")

    # Find asset
    target_name = f"ripgrep-{version}-{arch}-{os_tag}"
    asset = None
    for a in assets:
        if a["name"].startswith(target_name) and a["name"].endswith(ext):
            asset = a
            break
    if not asset:
        raise RuntimeError(
            f"No ripgrep asset found for {arch}-{os_tag} (version {version})"
        )

    asset_url = asset["browser_download_url"]
    asset_name = asset["name"]
    print(f"[ripgrep] Downloading {asset_name}...", file=sys.stderr)

    # Download with progress
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = Path(tmp_file.name)
            with urllib.request.urlopen(asset_url, timeout=30) as response:
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = downloaded * 100 // total
                        print(
                            f"\r[ripgrep] {percent}% downloaded",
                            end="",
                            file=sys.stderr,
                        )
                print(file=sys.stderr)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # Verify SHA256 (if available in release notes)
    sha256sum = None
    for line in release_info["body"].splitlines():
        if asset_name in line and "sha256" in line.lower():
            parts = line.split()
            for part in parts:
                if len(part) == 64 and all(
                    c in "0123456789abcdef" for c in part.lower()
                ):
                    sha256sum = part.lower()
                    break
    if sha256sum:
        print("[ripgrep] Verifying SHA256...", file=sys.stderr)
        hash_sha256 = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        if hash_sha256.hexdigest() != sha256sum:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("SHA256 verification failed!")

    # Extract binary
    print("[ripgrep] Extracting...", file=sys.stderr)
    try:
        if ext == ".zip":
            with zipfile.ZipFile(tmp_path, "r") as zf:
                # Find rg(.exe) inside zip (in subdir like "ripgrep-xx/")
                for member in zf.namelist():
                    if os.path.basename(member) == bin_file:
                        zf.extract(member, tmp_path.parent)
                        extracted = Path(tmp_path.parent) / member
                        break
                else:
                    raise RuntimeError(f"{bin_file} not found in zip")
        else:  # .tar.gz
            with tarfile.open(tmp_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if os.path.basename(member.name) == bin_file:
                        tf.extract(member, tmp_path.parent)
                        extracted = Path(tmp_path.parent) / member.name
                        break
                else:
                    raise RuntimeError(f"{bin_file} not found in tarball")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Extraction failed: {e}")

    # Move to final location
    try:
        shutil.move(str(extracted), final_bin)
        final_bin.chmod(0o755)  # make executable (Unix)
    except Exception as e:
        raise RuntimeError(f"Failed to install binary: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
        # Clean up extracted dir if any
        if extracted.parent != tmp_path.parent:
            shutil.rmtree(extracted.parent, ignore_errors=True)

    # Ensure install_dir in PATH (temporary for current process)
    str_install_dir = str(install_dir.resolve())
    if str_install_dir not in os.environ["PATH"]:
        os.environ["PATH"] = str_install_dir + os.pathsep + os.environ["PATH"]
        print(
            f"[ripgrep] Added {str_install_dir} to PATH (current session).",
            file=sys.stderr,
        )

    # Verify installation
    try:
        ver = subprocess.run(
            [str(final_bin), "--version"], capture_output=True, text=True, timeout=5
        )
        if ver.returncode == 0 and "ripgrep" in ver.stdout:
            print(f"[ripgrep] Successfully installed: {final_bin}", file=sys.stderr)
            return str(final_bin)
        else:
            raise RuntimeError(f"Verification failed: {ver.stderr}")
    except Exception as e:
        raise RuntimeError(f"Installation verification failed: {e}")


if __name__ == "__main__":
    try:
        rg_path = install_ripgrep()
        print(f"ripgrep installed at: {rg_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
