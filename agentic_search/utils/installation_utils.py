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


def install_rga(
    force_reinstall: bool = False,
    install_dir: Optional[str] = None,
    bin_name: str = "rga",
) -> str:
    """Automatically detect, download, and install ripgrep-all (rga) and its preprocessor.

    Installs both 'rga' and 'rga-preproc' to a user-local bin directory.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # 1. Normalize architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    # 2. Determine OS tag and file extension
    if system == "windows":
        os_tag = "pc-windows-msvc"
        ext = ".zip"
        required_bins = ["rga.exe", "rga-preproc.exe"]
    elif system == "darwin":
        os_tag = "apple-darwin"
        ext = ".tar.gz"
        required_bins = ["rga", "rga-preproc"]
    elif system == "linux":
        os_tag = "unknown-linux-musl"
        ext = ".tar.gz"
        required_bins = ["rga", "rga-preproc"]
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # 3. Set installation directory
    if install_dir is None:
        if system == "windows":
            install_dir = os.path.expandvars(r"%LOCALAPPDATA%\bin")
        else:
            install_dir = os.path.expanduser("~/.local/bin")

    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    # Path to the main binary for return value
    final_bin = install_dir / (bin_name + (".exe" if system == "windows" else ""))

    # 4. Check if already installed (Verify both rga and rga-preproc)
    if not force_reinstall:
        all_exist = all((install_dir / b).exists() for b in required_bins)
        if all_exist:
            try:
                ver = subprocess.run(
                    [str(final_bin), "--version"], capture_output=True, text=True
                )
                if ver.returncode == 0 and "ripgrep-all" in ver.stdout:
                    return str(final_bin)
            except Exception:
                pass

    # 5. Fetch latest version info
    print("[rga] Detecting latest version...", file=sys.stderr)
    try:
        api_url = "https://api.github.com/repos/phiresky/ripgrep-all/releases/latest"
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            release_info = json.loads(resp.read())
        assets = release_info["assets"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch rga release info: {e}")

    # 6. Find matching asset package
    asset = None
    for a in assets:
        name = a["name"]
        if arch in name and os_tag in name and name.endswith(ext):
            asset = a
            break

    if not asset:
        raise RuntimeError(f"No rga asset found for {arch}-{os_tag}")

    asset_url = asset["browser_download_url"]
    print(f"[rga] Downloading {asset['name']}...", file=sys.stderr)

    # 7. Download the archive
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = Path(tmp_file.name)
            with urllib.request.urlopen(asset_url, timeout=60) as response:
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                while True:
                    chunk = response.read(16384)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(
                            f"\r[rga] {downloaded * 100 // total}% downloaded",
                            end="",
                            file=sys.stderr,
                        )
                print(file=sys.stderr)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # 8 & 9. Extract and install ALL required binaries
    print(f"[rga] Extracting binaries to {install_dir}...", file=sys.stderr)
    temp_extract_dir = Path(tempfile.mkdtemp())
    try:
        if ext == ".zip":
            with zipfile.ZipFile(tmp_path, "r") as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename in required_bins:
                        extracted_path = zf.extract(member, temp_extract_dir)
                        target_path = install_dir / filename
                        shutil.move(extracted_path, target_path)
                        target_path.chmod(0o755)
        else:  # .tar.gz
            with tarfile.open(tmp_path, "r:gz") as tf:
                for member in tf.getmembers():
                    filename = os.path.basename(member.name)
                    if filename in required_bins:
                        tf.extract(member, temp_extract_dir)
                        extracted_path = temp_extract_dir / member.name
                        target_path = install_dir / filename
                        shutil.move(str(extracted_path), str(target_path))
                        target_path.chmod(0o755)
    except Exception as e:
        raise RuntimeError(f"Extraction/Installation failed: {e}")
    finally:
        # 10. Cleanup
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # 11. Verify installation
    try:
        subprocess.run([str(final_bin), "--version"], capture_output=True, check=True)
        # Verify preprocessor is also in the same directory
        if not (install_dir / required_bins[1]).exists():
            raise RuntimeError(f"Preprocessor {required_bins[1]} missing after install")

        print(
            f"[rga] Successfully installed rga and rga-preproc to: {install_dir}",
            file=sys.stderr,
        )
        return str(final_bin)
    except Exception as e:
        raise RuntimeError(f"Installation verification failed: {e}")


if __name__ == "__main__":
    try:
        path = install_rga()
        print(f"ripgrep-all is ready at: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
