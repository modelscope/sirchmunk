# Copyright (c) ModelScope Contributors. All rights reserved.
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import msgpack
from loguru import logger

from sirchmunk.schema.knowledge import KnowledgeCluster
from sirchmunk.utils.file_utils import StorageStructure

from .base import Storage

"""
High-performance Knowledge Cluster Store for large-scale graphs.

- Format: MessagePack + \x00 separator
- Index: id → (offset, length)
- Features: batch build, incremental insert, mmap read, auto-close
"""


class KnowledgeStorage(Storage):
    """
    Thread-safe store for KnowledgeCluster objects.

    Files layout:
        data_dir/
        ├── clusters.mpk   # MessagePack data: [packed]\x00[packed]\x00...
        └── clusters.idx   # Pickle: Dict[str, Tuple[int, int]]
    """

    def __init__(self, work_path: Union[str, Path], readonly: bool = False, **kwargs):

        self.work_path: Path = Path(work_path)

        self.mpk_path: Path = (
            self.work_path
            / StorageStructure.CACHE_DIR
            / StorageStructure.KNOWLEDGE_DIR
            / StorageStructure.CLUSTER_CONTENT_FILE
        )
        self.idx_path: Path = (
            self.work_path
            / StorageStructure.CACHE_DIR
            / StorageStructure.KNOWLEDGE_DIR
            / StorageStructure.CLUSTER_INDEX_FILE
        )
        self.readonly: bool = readonly

        super().__init__(self.idx_path, self.mpk_path, **kwargs)

    # ======================
    # WRITE OPERATIONS (thread-safe)
    # ======================

    def build_from_clusters(
        self,
        clusters: Union[List[Dict[str, Any]], List[KnowledgeCluster]],
        batch_size: int = 10_000,
        overwrite: bool = True,
    ) -> None:
        """
        Build store from a list of KnowledgeCluster objects.

        Args:
            clusters: List of KnowledgeCluster objects
            batch_size: Number of clusters to write per batch
            overwrite: If True, overwrite existing store; else append

        Returns:
            None
        """
        if self.readonly:
            raise PermissionError("Store is readonly")

        with self._index_lock:
            if overwrite or not self.mpk_path.exists():
                self._offsets.clear()
                self.mpk_path.unlink(missing_ok=True)

            self._ensure_dir()
            current_offset = (
                self.mpk_path.stat().st_size if self.mpk_path.exists() else 0
            )

            with open(self.mpk_path, "ab") as f_mpk:
                for i in range(0, len(clusters), batch_size):
                    batch = clusters[i : i + batch_size]
                    for cluster in batch:
                        data = (
                            cluster if isinstance(cluster, dict) else cluster.to_dict()
                        )
                        cluster_id = data.get("id")
                        if not cluster_id:
                            raise ValueError("Cluster must have 'id' in to_dict()")

                        packed = msgpack.packb(data, use_bin_type=True)
                        record = packed + b"\x00"

                        self._offsets[cluster_id] = (current_offset, len(record))
                        current_offset += len(record)

                        f_mpk.write(record)
                        f_mpk.flush()

            self._save_index()

            # Ensure mmap reflects latest file content
            self._refresh_mmap()

    def insert_cluster(self, cluster: Any, overwrite: bool = True) -> None:
        """Insert a single cluster (append-only)."""
        self.build_from_clusters([cluster], batch_size=1, overwrite=overwrite)

    def insert_clusters(self, clusters: List[Any], overwrite: bool = True) -> None:
        """Insert multiple clusters."""
        self.build_from_clusters(clusters, batch_size=1000, overwrite=overwrite)

    # ======================
    # READ OPERATIONS
    # ======================

    def get(self, cluster_id: str) -> dict:
        """O(1) random access by cluster_id."""
        with self._index_lock:
            if cluster_id not in self._offsets:
                raise KeyError(f"Cluster '{cluster_id}' not found")

            start, length = self._offsets[cluster_id]

        # Access mmap outside lock for concurrency
        if self._mmap is None:
            self._open_mmap()
            if self._mmap is None:
                raise RuntimeError("Data file not available")

        end = start + length
        if end > self._mmap.size():
            raise RuntimeError(f"Index corruption: {cluster_id} out of bounds")

        raw_bytes = self._mmap[start : end - 1]  # exclude \x00
        return msgpack.unpackb(raw_bytes, raw=False, strict_map_key=False)

    def get_batch(self, cluster_ids: List[str]) -> List[dict]:
        """Batch get for better cache locality."""
        return [self.get(cid) for cid in cluster_ids]

    def keys(self) -> Iterator[str]:
        """Iterator over all cluster IDs."""
        return iter(self._offsets.keys())

    def items(self) -> Iterator[Tuple[str, dict]]:
        """Iterator over (id, cluster_dict)."""
        for cid in self.keys():
            yield cid, self.get(cid)

    def rebuild(self) -> None:
        """
        Rebuild storage by rewriting only logically live clusters.
        Removes dead entries, reclaims disk space, and improves data locality.
        """
        if self.readonly:
            raise PermissionError("Storage is readonly")

        logger.info("Starting storage rebuild...")

        # Step 1: Collect all live cluster dicts BEFORE closing resources
        live_clusters: List[Dict[str, Any]] = []
        with self._index_lock:
            # Snapshot current live IDs (thread-safe)
            live_ids = list(self._offsets.keys())
            for cid in live_ids:
                try:
                    data = self.get(cid)
                    # Ensure 'id' field is present (critical for indexing)
                    if "id" not in data:
                        logger.warning(f"Cluster {cid} missing 'id'; skipping")
                        continue
                    live_clusters.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read cluster {cid} during rebuild: {e}")

        logger.info(f"Collected {len(live_clusters)} live clusters for rebuild.")

        # Step 2: Close current file handles to release resources
        self.close()

        # Step 3: Remove old files
        self.mpk_path.unlink(missing_ok=True)
        self.idx_path.unlink(missing_ok=True)

        # Step 4: Recreate directory if needed
        self._ensure_dir()

        # Step 5: Rebuild index and data using dict list — no mocking needed
        self._offsets.clear()  # reset in-memory index
        self.build_from_clusters(
            clusters=live_clusters, batch_size=10_000, overwrite=True
        )

        logger.info(f"Rebuild completed. {len(live_clusters)} clusters retained.")


# ======================
# Utility: Index Merge (for distributed build)
# ======================


def merge_stores(
    store_paths: List[str],
    output_dir: str,
    batch_size: int = 10_000,
) -> None:
    """
    Merge multiple KnowledgeStore directories into one.

    Use case:
        - Build shards in parallel: store_0/, store_1/, ...
        - Merge into final store: merged/

    Args:
        store_paths: List of paths to store directories
        output_dir: Output directory
        batch_size: Batch size for writing
    """
    output_store = KnowledgeStorage(output_dir)
    output_store._ensure_dir()

    # Clear target files
    output_store.mpk_path.unlink(missing_ok=True)
    output_store.idx_path.unlink(missing_ok=True)

    total_written = 0
    current_offset = 0

    with open(output_store.mpk_path, "wb") as f_out:
        for store_path in store_paths:
            store = KnowledgeStorage(store_path, readonly=True)
            logger.info(f"Merging {store_path} ({len(store)} clusters)...")

            # Stream clusters in batches
            ids = list(store.keys())
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_data = [store.get(cid) for cid in batch_ids]

                for data in batch_data:
                    cluster_id = data["id"]
                    packed = msgpack.packb(data, use_bin_type=True)
                    record = packed + b"\x00"

                    # Write data
                    f_out.write(record)
                    f_out.flush()

                    # Update index
                    output_store._offsets[cluster_id] = (current_offset, len(record))
                    current_offset += len(record)

                total_written += len(batch_data)

            store.close()

    # Save final index
    output_store._save_index()
    logger.info(f"Merged {total_written} clusters into {output_dir}")
