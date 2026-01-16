# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger
from tqdm.auto import tqdm

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.scan.base import BaseScanner
from sirchmunk.schema.metadata import FileInfo, build_file_schema
from sirchmunk.utils.file_utils import StorageStructure

METADATA_NAME = ".metadata"


class FileScanner(BaseScanner):
    """High-performance file metadata scanner with incremental batch processing.

    Scans files under specified corpus paths, generates metadata using schema builders,
    skips unchanged files, and persists results incrementally in shuffled batches.
    Utilizes thread pooling for concurrency and efficient change detection.

    Attributes:
        corpus_paths (List[Path]): Paths to scan (directories or individual files).
        work_path (Path): Base directory for operations (defaults to current working directory).
        metadata_path (Path): Directory to store metadata JSON files (under work_path).
        max_workers (int): Maximum thread pool workers for concurrent scanning.
        batch_size (int): Number of files processed before saving metadata (default: 1000).
        _base_metadata_cache (Dict[Path, Tuple[int, datetime]]): Cache of existing file stats
            for change detection (size in bytes and last modification time).
    """

    def __init__(
        self,
        corpus_path: Union[str, Path, List[str], List[Path]],
        llm: Optional[OpenAIChat] = None,
        work_path: Union[str, Path, None] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 1000,
        verbose: bool = False,
    ):
        """Initialize the file scanner.

        Args:
            corpus_path: Single or multiple paths to scan (files or directories).
            work_path: Base directory for metadata storage. Defaults to current directory.
            max_workers: Maximum threads for concurrent scanning. Defaults to
                min(32, CPU_COUNT * 2) if unset.
            batch_size: Number of files to process before saving metadata. Defaults to 1000.
        """
        # Normalize corpus paths to a list of Path objects
        if isinstance(corpus_path, (str, Path)):
            corpus_path = [corpus_path]
        self.corpus_paths: List[Path] = [Path(p).resolve() for p in corpus_path]

        # Set work and metadata paths
        self.work_path: Path = (
            Path.cwd() if work_path is None else Path(work_path).resolve()
        )
        self.metadata_path: Path = (
            self.work_path / StorageStructure.CACHE_DIR / StorageStructure.METADATA_DIR
        )

        # Configure thread pool size
        cpu_count = os.cpu_count() or 1
        self.max_workers: int = max_workers or min(32, cpu_count * 2)
        self.batch_size: int = batch_size

        # Cache for existing file stats to detect changes
        self._base_metadata_cache: Dict[str, Any] = {}
        self._base_metadata_cache_paths: Set[str] = (
            set()
        )  # `/path/to/xxx.ext@ISOtimestamp`

        # Ensure metadata directory exists
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        self.verbose = verbose

        self.llm = llm

        super().__init__()

    def scan(
        self,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        tqdm_desc: str = "Scanning files",
    ) -> List[Any]:
        """Scan files and generate metadata in shuffled batches with incremental saving and progress tracking.

        Skips files unchanged since last scan (based on abs path and modification time).
        Uses thread pooling for concurrent processing within batches. Saves metadata
        after each batch completes to ensure progress persistence.

        Args:
            max_workers: Override default thread count for this scan.
            batch_size: Override default batch size. Must be >= 1.
            shuffle: Whether to shuffle files before batching (improves I/O distribution).
            tqdm_desc: Description prefix for the progress bar.
            llm: The OpenAI api format client.

        Returns:
            List of metadata objects (FileInfo or subclasses) for all scanned files.
        """
        effective_batch_size = batch_size or self.batch_size
        if effective_batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        # Load existing metadata stats for change detection
        self._load_base_metadata_cache()

        # Collect all file paths to consider (skip directories and symlinks)
        all_files: List[Path] = []
        for path in self.corpus_paths:
            if path.is_dir():
                all_files.extend(
                    p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
                )
            elif path.is_file():
                all_files.append(path.resolve())

        # Quick filter
        files_to_scan = [f for f in all_files if not self._should_exclude(f)]

        total_files = len(files_to_scan)
        total_skipped = len(all_files) - total_files

        # Shuffle files to distribute I/O load evenly across batches
        if shuffle:
            random.shuffle(files_to_scan)

        logger.info(
            f"Scanning {total_files} of {len(all_files)} files "
            f"(skipped {total_skipped} unchanged) in "
            f"{(total_files // effective_batch_size) + (1 if total_files % effective_batch_size else 0)} batches"
        )

        if total_files == 0:
            logger.info("No new or modified files to scan.")
            return []

        # Prepare batches
        batches = [
            files_to_scan[i : i + effective_batch_size]
            for i in range(0, len(files_to_scan), effective_batch_size)
        ]
        total_batches = len(batches)

        # Initialize progress bar
        pbar = tqdm(
            total=total_files,
            desc=tqdm_desc,
            unit="file",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        total_results = []
        total_success = 0
        total_failed = 0

        try:
            for batch_idx, batch in enumerate(batches, 1):
                batch_size_actual = len(batch)
                # Update progress bar description for current batch
                pbar.set_postfix_str(
                    f"Batch {batch_idx}/{total_batches}, "
                    f"Success: {total_success}, Failed: {total_failed}"
                )

                batch_results = self._process_batch(batch, max_workers, self.llm)
                success_in_batch = len(batch_results)
                failed_in_batch = batch_size_actual - success_in_batch

                # Increment counters
                total_success += success_in_batch
                total_failed += failed_in_batch

                # Save and aggregate
                self.save(batch_results)
                total_results.extend(batch_results)

                # Update progress bar by actual processed count (not just batch size)
                pbar.update(batch_size_actual)

                # Update postfix with fresh stats
                success_rate = (
                    (total_success / (total_success + total_failed)) * 100
                    if (total_success + total_failed) > 0
                    else 0.0
                )
                pbar.set_postfix_str(
                    f"B{batch_idx}/{total_batches}, "
                    f"✓{total_success} ✗{total_failed} "
                    f"({success_rate:.1f}% ok)"
                )

            pbar.set_postfix_str(f"✓{total_success} ✗{total_failed} (done)")
        finally:
            pbar.close()

        logger.info(
            f"Scan completed: {total_success} succeeded, {total_failed} failed, "
            f"{total_skipped} skipped (unchanged)."
        )
        return total_results

    def save(self, metadata_list: List[FileInfo]) -> None:
        """Persist metadata objects incrementally to disk.

        Saves each metadata object as a separate JSON file in metadata_path.
        Filenames are SHA-256 hashes of absolute file paths to avoid collisions.
        Only new/updated files are written; existing unchanged files are untouched.

        Args:
            metadata_list: List of metadata objects (FileInfo or subclasses) to save.
        """
        for metadata in metadata_list:
            try:
                # Convert metadata object to JSON-serializable dictionary
                data = self._serialize_metadata(metadata)
                if not metadata.cache_key:
                    continue
                output_file = self.metadata_path / f"{metadata.cache_key}.json"

                # Write JSON file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                if self.verbose:
                    logger.debug(
                        f"Saved metadata for {metadata.file_or_url} to {output_file}"
                    )
            except Exception as e:
                logger.error(f"Failed to save metadata for {metadata.file_or_url}: {e}")

    def load(self) -> Dict[Path, Any]:
        """Load all existing metadata from disk into a path-indexed dictionary.

        Reads all JSON files in metadata_path, deserializes them, and maps by file path.
        Intended for external use (e.g., reporting); not used internally for scanning.

        Returns:
            Dictionary mapping file paths to their metadata objects.
        """
        metadata_map = {}
        for meta_file in self.metadata_path.glob("*.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Deserialize JSON to metadata object (basic reconstruction)
                path = Path(data["path"])
                metadata_map[path] = data
            except Exception as e:
                logger.warning(f"Error loading metadata file {meta_file}: {e}")
        return metadata_map

    def _load_base_metadata_cache(self) -> None:
        """Populate internal cache with size and mtime of existing metadata files.

        Enables efficient change detection during scanning by avoiding full metadata loads.
        Cache format: {file_path: (size_bytes, last_modified_datetime)}
        """
        self._base_metadata_cache.clear()
        self._base_metadata_cache_paths.clear()
        for meta_file in self.metadata_path.glob("*.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    file_info: Dict[str, Any] = json.load(f)

                file_info: FileInfo = FileInfo.from_dict(file_info)
                self._base_metadata_cache[file_info.cache_key] = file_info
                self._base_metadata_cache_paths.add(
                    FileInfo.get_path_mtime(
                        file_info.file_or_url, file_info.last_modified
                    )
                )
            except (KeyError, ValueError, TypeError, OSError) as e:
                if self.verbose:
                    logger.warning(f"Invalid metadata in {meta_file}: {e}")

    def _process_batch(
        self,
        file_batch: List[Path],
        max_workers: Optional[int] = None,
        llm: Optional[OpenAIChat] = None,
    ) -> List[Any]:
        """Process a batch of files concurrently and return successful metadata objects.

        Args:
            file_batch: List of file paths to process in this batch.
            max_workers: Override default thread count for this batch.

        Returns:
            List of metadata objects (FileInfo or subclasses) for successfully processed files.
        """
        results = []
        workers = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_file, file_path, llm): file_path
                for file_path in file_batch
            }
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    metadata = future.result(timeout=60)  # 60-second timeout per file
                    if metadata is not None:
                        results.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to process {file_path} in batch: {e}")
        return results

    def _process_file(
        self,
        file_path: Path,
        llm: Optional[OpenAIChat] = None,
    ) -> Optional[Any]:
        """Process a single file to generate its metadata schema.

        Wraps build_file_schema with error handling and logging.

        Args:
            file_path: Path to the file to process.

        Returns:
            Metadata object (FileInfo or subclass) if successful; None on failure.
        """
        try:
            if self.verbose:
                logger.debug(f"Processing file: {file_path}")
            return build_file_schema(
                path=file_path,
                llm=llm,
            )
        except Exception as e:
            logger.error(f"Schema build failed for {file_path}: {e}")
            return None

    def _serialize_metadata(self, metadata: FileInfo) -> Dict[str, Any]:
        """Convert a metadata object to a JSON-serializable dictionary.

        Handles Path and datetime conversions, and recursively processes dataclasses.

        Args:
            metadata: Metadata object (instance of FileInfo or subclass).

        Returns:
            Dictionary ready for JSON serialization.
        """
        return metadata.to_dict()

    def _should_exclude(self, f: Path) -> bool:
        """
        Quick check if the file should be excluded from scanning based on existing cache.
            key: `/path/to/xxx.ext@ISOtimestamp`

        Args:
            f: Path object of the file to check.

        Returns:
            bool: True if the file should be excluded (unchanged), False otherwise.
        """
        stat = f.stat()
        mtime: datetime = datetime.fromtimestamp(stat.st_mtime)

        return FileInfo.get_path_mtime(f, mtime) in self._base_metadata_cache_paths
