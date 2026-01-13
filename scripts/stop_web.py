#!/usr/bin/env python3
"""
Stop script for DeepTutor Web Platform
Terminates all backend and frontend processes to free up ports
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def print_flush(*args, **kwargs):
    """Print with flush=True by default"""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def get_backend_port():
    """Get backend port from environment or use default"""
    return int(os.environ.get("BACKEND_PORT", os.environ.get("API_PORT", "8001")))


def get_frontend_port():
    """Get frontend port from environment or use default"""
    return int(os.environ.get("FRONTEND_PORT", os.environ.get("PORT", "3782")))


def find_processes_by_port(port):
    """Find processes using a specific port"""
    processes = []
    try:
        if os.name == "nt":
            # Windows: Use netstat to find processes
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            processes.append(int(pid))
                        except ValueError:
                            continue
        else:
            # Unix: Use lsof to find processes
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        try:
                            processes.append(int(line.strip()))
                        except ValueError:
                            continue
    except FileNotFoundError:
        # lsof or netstat not available, try alternative methods
        pass

    return processes


def find_processes_by_name(name_patterns):
    """Find processes by name patterns"""
    processes = []
    try:
        if os.name == "nt":
            # Windows: Use tasklist
            result = subprocess.run(
                ["tasklist", "/fo", "csv"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.split("\n")[1:]:  # Skip header
                if line.strip():
                    parts = line.split('","')
                    if len(parts) >= 2:
                        process_name = parts[0].strip('"')
                        pid_str = parts[1].strip('"')
                        try:
                            pid = int(pid_str)
                            if any(
                                pattern.lower() in process_name.lower()
                                for pattern in name_patterns
                            ):
                                processes.append(pid)
                        except ValueError:
                            continue
        else:
            # Unix: Use pgrep
            for pattern in name_patterns:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            try:
                                processes.append(int(line.strip()))
                            except ValueError:
                                continue
    except FileNotFoundError:
        pass

    return processes


def kill_process(pid, name="Process", force=False):
    """Kill a process by PID"""
    try:
        if os.name == "nt":
            # Windows: Use taskkill
            cmd = ["taskkill", "/PID", str(pid)]
            if force:
                cmd.extend(["/F", "/T"])  # Force kill with tree
            else:
                cmd.append("/T")  # Kill tree gracefully

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print_flush(f"   ✅ Killed {name} (PID: {pid})")
                return True
            else:
                if "not found" not in result.stderr.lower():
                    print_flush(
                        f"   ⚠️ Failed to kill {name} (PID: {pid}): {result.stderr.strip()}"
                    )
                return False
        else:
            # Unix: Use kill
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            print_flush(
                f"   ✅ Killed {name} (PID: {pid}) with {'SIGKILL' if force else 'SIGTERM'}"
            )
            return True
    except ProcessLookupError:
        # Process already terminated
        return True
    except PermissionError:
        print_flush(f"   ⚠️ Permission denied to kill {name} (PID: {pid})")
        return False
    except Exception as e:
        print_flush(f"   ⚠️ Error killing {name} (PID: {pid}): {e}")
        return False


def stop_processes_by_port(port, service_name):
    """Stop all processes using a specific port"""
    print_flush(f"🔍 Looking for {service_name} processes on port {port}...")

    processes = find_processes_by_port(port)
    if not processes:
        print_flush(f"   ✅ No processes found on port {port}")
        return True

    print_flush(f"   Found {len(processes)} process(es) on port {port}: {processes}")

    # First try graceful termination
    killed_count = 0
    for pid in processes:
        if kill_process(pid, f"{service_name}", force=False):
            killed_count += 1

    if killed_count > 0:
        print_flush("   ⏳ Waiting 3 seconds for graceful shutdown...")
        time.sleep(3)

        # Check if any processes are still running
        remaining = find_processes_by_port(port)
        if remaining:
            print_flush(f"   🔄 Force killing remaining processes: {remaining}")
            for pid in remaining:
                kill_process(pid, f"{service_name}", force=True)

    # Final check
    final_check = find_processes_by_port(port)
    if final_check:
        print_flush(
            f"   ⚠️ Some processes may still be running on port {port}: {final_check}"
        )
        return False
    else:
        print_flush(f"   ✅ Port {port} is now free")
        return True


def stop_processes_by_name(patterns, service_name):
    """Stop processes by name patterns"""
    print_flush(f"🔍 Looking for {service_name} processes by name...")

    processes = find_processes_by_name(patterns)
    if not processes:
        print_flush(f"   ✅ No {service_name} processes found")
        return True

    print_flush(f"   Found {len(processes)} {service_name} process(es): {processes}")

    # First try graceful termination
    killed_count = 0
    for pid in processes:
        if kill_process(pid, f"{service_name}", force=False):
            killed_count += 1

    if killed_count > 0:
        print_flush("   ⏳ Waiting 2 seconds for graceful shutdown...")
        time.sleep(2)

        # Check if any processes are still running
        remaining = find_processes_by_name(patterns)
        if remaining:
            print_flush(f"   🔄 Force killing remaining processes: {remaining}")
            for pid in remaining:
                kill_process(pid, f"{service_name}", force=True)

    return True


def main():
    print_flush("=" * 50)
    print_flush("DeepTutor Web Platform Stopper")
    print_flush("=" * 50)

    # Load environment variables from .env files
    project_root = Path(__file__).parent.parent
    env_files = [project_root / "DeepTutor.env", project_root / ".env"]

    for env_file in env_files:
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=False)
                print_flush(f"✅ Loaded environment from {env_file}")
                break
            except ImportError:
                print_flush("⚠️ python-dotenv not installed, using default ports")
                break

    backend_port = get_backend_port()
    frontend_port = get_frontend_port()

    print_flush(f"🎯 Target ports - Backend: {backend_port}, Frontend: {frontend_port}")
    print_flush("")

    success = True

    # Stop backend processes
    print_flush("🛑 Stopping Backend Services...")
    if not stop_processes_by_port(backend_port, "Backend"):
        success = False

    # Also try to stop by process name patterns
    backend_patterns = ["uvicorn", "src.api.main", "fastapi", "python.*api.*main"]
    stop_processes_by_name(backend_patterns, "Backend")

    print_flush("")

    # Stop frontend processes
    print_flush("🛑 Stopping Frontend Services...")
    if not stop_processes_by_port(frontend_port, "Frontend"):
        success = False

    # Also try to stop by process name patterns
    frontend_patterns = ["next-server", "next dev", "npm.*dev", "node.*next"]
    stop_processes_by_name(frontend_patterns, "Frontend")

    print_flush("")

    # Clean up any remaining Node.js processes that might be related
    print_flush("🧹 Cleaning up related processes...")
    cleanup_patterns = [
        "node.*3782",  # Frontend port specific
        f"node.*{frontend_port}",  # Dynamic frontend port
        "webpack",
        "turbopack",
    ]
    stop_processes_by_name(cleanup_patterns, "Related")

    print_flush("")
    print_flush("=" * 50)
    if success:
        print_flush("✅ All services stopped successfully!")
        print_flush(
            f"   Ports {backend_port} and {frontend_port} should now be available."
        )
    else:
        print_flush("⚠️ Some processes may still be running.")
        print_flush(
            "   You may need to restart your terminal or reboot if ports remain occupied."
        )
    print_flush("=" * 50)

    # Final port check
    print_flush("")
    print_flush("🔍 Final port availability check...")

    import socket

    for port, name in [(backend_port, "Backend"), (frontend_port, "Frontend")]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                print_flush(f"   ⚠️ Port {port} ({name}) is still in use")
            else:
                print_flush(f"   ✅ Port {port} ({name}) is available")
        except Exception as e:
            print_flush(f"   ❓ Could not check port {port} ({name}): {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_flush("\n🛑 Stop operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_flush(f"\n❌ Error during stop operation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
