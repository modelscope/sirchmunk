import hashlib
import os
from pathlib import Path
from typing import Union

from loguru import logger


def get_fast_hash(file_path: Union[str, Path], sample_size: int = 8192):
    """
    Computes a partial hash (fingerprint) by combining:
    File Size + Head Chunk + Tail Chunk.
    This is extremely efficient for large-scale file hash calculation.
    """
    file_path = Path(file_path)
    try:
        # Get metadata first (O(1) operation)
        file_size = file_path.stat().st_size

        # If the file is smaller than the combined sample size, read it entirely
        if file_size <= sample_size * 2:
            with open(file_path, "rb") as f:
                return f"{hashlib.md5(f.read()).hexdigest()}_{file_size}"

        # Large file sampling: Read head and tail to avoid full disk I/O
        hash_content = hashlib.md5()
        with open(file_path, "rb") as f:
            hash_content.update(f.read(sample_size))
            f.seek(-sample_size, os.SEEK_END)
            hash_content.update(f.read(sample_size))

        # Mix the file size into the hash string to minimize collisions
        return f"{hash_content.hexdigest()}_{file_size}"
    except (FileNotFoundError, PermissionError):
        # Handle cases where files are deleted during scan or access is denied
        logger.warning("File not found or inaccessible: {}", file_path)
        return None


class StorageStructure:
    """
    Standardized directory and file naming conventions for caching and storage.
    """

    CACHE_DIR = ".cache"

    METADATA_DIR = "metadata"

    KNOWLEDGE_DIR = "knowledge"

    COGNITION_DIR = "cognition"

    # `.idx` -> Index file for fast lookup of cluster content
    CLUSTER_INDEX_FILE = "cluster.idx"

    # `.mpk` -> MessagePack serialized cluster content
    CLUSTER_CONTENT_FILE = "cluster.mpk"
