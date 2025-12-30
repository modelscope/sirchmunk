import mmap
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import msgpack
from loguru import logger


class Storage:
    """
    Base class for storage systems based on mmap.
    """

    def __init__(
        self,
        idx_path: Union[str, Path],
        mpk_path: Union[str, Path],
        readonly: bool = False,
        **kwargs,
    ):

        self.idx_path = Path(idx_path)
        self.mpk_path = Path(mpk_path)
        self.readonly = readonly

        self._kwargs = kwargs

        # Thread safety for index updates
        self._index_lock = threading.RLock()
        self._offsets: Dict[str, Tuple[int, int]] = {}
        self._mmap: Optional[mmap.mmap] = None
        self._file: Optional[Any] = None

        # Load index if exists
        if self.idx_path.exists():
            self._load_index()
        else:
            self._offsets = {}

        # Open mmap only for reading
        if self.mpk_path.exists():
            self._open_mmap()

    def _load_index(self):
        """Load index from disk."""
        try:
            with open(self.idx_path, "rb") as f:
                self._offsets = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Failed to load index: {e}. Rebuilding...")
            self._offsets = {}

    def _open_mmap(self):
        """Open read-only mmap."""
        if self._mmap is not None:
            return
        try:
            self._file = open(self.mpk_path, "rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to mmap {self.mpk_path}: {e}")
            self._mmap = None

    def _ensure_dir(self):
        """Ensure data directory exists."""
        knowledge_dir: Path = self.idx_path.parent.resolve()
        knowledge_dir.mkdir(parents=True, exist_ok=True)

    def _save_index(self):
        """Save index to disk (atomic rename to avoid corruption)."""
        if self.readonly:
            return

        tmp_path = self.idx_path.with_suffix(".idx.tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self._offsets, f)
            os.replace(tmp_path, self.idx_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to save index: {e}")

    def _refresh_mmap(self):
        """Reopen mmap to capture file growth after writes."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self.mpk_path.exists():
            self._open_mmap()

    def __contains__(self, uid: str) -> bool:
        return uid in self._offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def repair(self):
        """Rebuild index by scanning the .mpk file (slow, for recovery only)."""
        if not self.mpk_path.exists():
            self._offsets.clear()
            self._save_index()
            return

        logger.info("Repairing index by scanning .mpk file ...")
        self._offsets.clear()
        offset = 0
        count = 0

        with open(self.mpk_path, "rb") as f:
            while True:
                # Read until \x00
                buf = bytearray()
                while True:
                    byte = f.read(1)
                    if not byte:
                        break
                    if byte == b"\x00":
                        break
                    buf.extend(byte)
                if not buf:
                    break

                try:
                    data = msgpack.unpackb(buf, raw=False)
                    cid = data.get("id")
                    if cid:
                        self._offsets[cid] = (offset, len(buf) + 1)  # +1 for \x00
                        count += 1
                except Exception as e:
                    logger.warning(f"Skip invalid record at {offset}: {e}")

                offset = f.tell()

        self._save_index()
        logger.info(f"Repaired: {count} samples indexed.")

    def delete_batch(self, uids: List[str]) -> int:
        """Delete multiple unique ids (index ONLY). Returns number of actually deleted."""
        if self.readonly:
            raise PermissionError("Storage is readonly")

        deleted = 0
        with self._index_lock:
            for _uid in uids:
                if _uid in self._offsets:
                    del self._offsets[_uid]
                    deleted += 1
            if deleted > 0:
                self._save_index()
        return deleted

    def close(self):
        """Close resources."""
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
