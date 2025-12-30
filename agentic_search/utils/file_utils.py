import hashlib
from pathlib import Path
from typing import Union


def md5_head(filepath: Union[str, Path], head_size: int = 1024 * 1024) -> str:
    """
    Calculate the MD5 hash of the first `head_size` bytes of a file for quick identification.

    Args:
        filepath (Union[str, Path]): The path to the file.
        head_size (int): The number of bytes to read from the start of the file. Default is 1MB.

    Returns:
        str: The MD5 hash of the file's head as a hexadecimal string.
    """
    hasher = hashlib.md5(usedforsecurity=False)
    with open(filepath, "rb") as f:
        hasher.update(f.read(head_size))
    return hasher.hexdigest()


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
