"""Fixed signed-source entry; safe to import in multiprocessing spawn children."""
from pathlib import Path
import sys


def main():
    # -I intentionally ignores PYTHONDONTWRITEBYTECODE. Set the interpreter
    # flag directly so a supervised process never writes into its code slot.
    sys.dont_write_bytecode = True
    # Resolve only this installed source slot, never cwd or a remote argv path.
    source = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source))
    from sirchmunk.api.supervisor import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
