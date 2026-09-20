# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger

from ..utils.constants import (
    DEFAULT_SIRCHMUNK_WORK_PATH,
    GREP_CONCURRENT_LIMIT,
    GREP_FALLBACK_CONCURRENT_LIMIT,
    GREP_FALLBACK_TIMEOUT,
    GREP_FALLBACK_TO_RG,
    GREP_KEYWORD_CONCURRENT_LIMIT,
    GREP_MAX_FILESIZE_MB,
    GREP_MAX_MATCHES_PER_FILE,
    GREP_PROCESS_KILL_TIMEOUT,
    GREP_QUEUE_TIMEOUT,
    GREP_RGA_ADAPTERS,
    GREP_RGA_BACKOFF_SECONDS,
    GREP_RICH_EXTENSIONS,
    GREP_TEXT_TIMEOUT,
    GREP_TIERED_SCAN,
    GREP_TIMEOUT,
)
from ..utils.file_utils import StorageStructure, fast_extract
from .base import BaseRetriever

RGA_SEMAPHORE = asyncio.Semaphore(value=GREP_CONCURRENT_LIMIT)
RG_FALLBACK_SEMAPHORE = asyncio.Semaphore(value=GREP_FALLBACK_CONCURRENT_LIMIT)
_RGA_BACKOFF_UNTIL = 0.0


class SearchProcessTimeoutError(RuntimeError):
    """Raised when a search subprocess cannot start or finish within its budget."""

    def __init__(self, command: str, stage: str, timeout: float) -> None:
        self.command = command
        self.stage = stage
        self.timeout = timeout
        super().__init__(f"{command} {stage} timed out ({timeout}s).")


class SearchExecutableNotFoundError(RuntimeError):
    """Raised when a configured search executable is unavailable."""

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(f"Search executable '{command}' was not found.")


class GrepRetriever(BaseRetriever):
    """A Python wrapper for ripgrep-all (rga), exposing its functionality via static methods.

    All methods are static and return parsed results. JSON output is preferred where possible
    for reliable parsing. Shell injection is mitigated by using `subprocess.run` with `shell=False`
    and explicit argument lists.

    For more information about ripgrep-all, please refer to `https://github.com/phiresky/ripgrep-all`
    """

    def __init__(self, work_path: Union[str, Path] = None, **kwargs):
        super().__init__()

        self.work_path: Path = Path(work_path or DEFAULT_SIRCHMUNK_WORK_PATH).expanduser().resolve()
        self.rga_cache: Path = (
            self.work_path / StorageStructure.CACHE_DIR / StorageStructure.GREP_DIR
        )
        self.rga_cache.mkdir(parents=True, exist_ok=True)
        self._native_text_cache: Dict[str, Tuple[int, int, str]] = {}

    async def retrieve_native_formats(
        self,
        terms: List[str],
        path: Union[str, Path, List[str], List[Path], None] = None,
        *,
        max_depth: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_results: int = 50,
        max_lines: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search formats that frequently fail rga preprocessing.

        The fallback is exact-term oriented and only activates for discriminative
        terms (digits, hyphens, or multi-word phrases).  Extracted text is cached
        by file size and nanosecond mtime for warm-query reuse.
        """
        if os.getenv("SIRCHMUNK_NATIVE_FORMAT_FALLBACK", "true").lower() != "true":
            return []
        precise_terms = [
            str(term).strip() for term in terms
            if str(term).strip() and (
                any(char.isdigit() for char in str(term))
                or "-" in str(term)
                or " " in str(term).strip()
            )
        ]
        if not precise_terms:
            return []
        roots = path if isinstance(path, list) else [path or Path.cwd()]
        target_extensions = {".log", ".xlsx", ".xls", ".pptx", ".ppt"}
        excluded = set(exclude or [])
        candidates: List[Path] = []
        for raw_root in roots:
            root = Path(raw_root).expanduser().resolve()
            if root.is_file():
                discovered = [root]
            elif root.is_dir():
                discovered = root.rglob("*")
            else:
                continue
            for file_path in discovered:
                if not file_path.is_file() or file_path.suffix.lower() not in target_extensions:
                    continue
                if any(file_path.match(pattern) for pattern in excluded):
                    continue
                if include and not any(file_path.match(pattern) for pattern in include):
                    continue
                if max_depth is not None and root.is_dir():
                    try:
                        if len(file_path.relative_to(root).parts) - 1 > max_depth:
                            continue
                    except ValueError:
                        continue
                candidates.append(file_path)

        semaphore = asyncio.Semaphore(4)

        async def _extract(
            file_path: Path,
        ) -> Optional[Tuple[str, str, bool, float]]:
            async with semaphore:
                started_at = time.perf_counter()
                try:
                    stat = file_path.stat()
                    key = str(file_path)
                    cached = self._native_text_cache.get(key)
                    if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
                        return key, cached[2], True, (
                            time.perf_counter() - started_at
                        ) * 1000
                    output = await fast_extract(file_path)
                    content = output.content or ""
                    self._native_text_cache[key] = (
                        stat.st_size, stat.st_mtime_ns, content,
                    )
                    return key, content, False, (
                        time.perf_counter() - started_at
                    ) * 1000
                except Exception as exc:
                    logger.debug(
                        "[native_format_search] extraction failed for %s: %s",
                        file_path, exc,
                    )
                    return None

        extracted = await asyncio.gather(
            *[_extract(file_path) for file_path in candidates],
        )
        results: List[Dict[str, Any]] = []
        lower_terms = [(term, term.lower()) for term in precise_terms]
        for extracted_item in extracted:
            if extracted_item is None:
                continue
            file_path, content, cache_hit, extraction_ms = extracted_item
            lines: List[str] = []
            matched_terms: List[str] = []
            for line in content.splitlines():
                lowered = line.lower()
                line_terms = [original for original, term in lower_terms if term in lowered]
                if not line_terms:
                    continue
                for term in line_terms:
                    if term not in matched_terms:
                        matched_terms.append(term)
                text = line.strip()
                if text and text not in lines:
                    lines.append(text)
                if len(lines) >= max_lines:
                    break
            if not lines:
                continue
            matches = [
                {
                    "type": "match",
                    "data": {"lines": {"text": line}},
                    "_keyword": matched_terms[0] if matched_terms else precise_terms[0],
                    "_search_backend": "native_extract",
                    "_search_cache_hit": cache_hit,
                    "_conversion_elapsed_ms": round(extraction_ms, 3),
                }
                for line in lines
            ]
            results.append({
                "path": file_path,
                "matches": matches,
                "lines": lines,
                "total_matches": len(lines),
                "total_score": float(len(matched_terms)),
            })
        results.sort(
            key=lambda item: (-item["total_score"], -item["total_matches"], item["path"]),
        )
        return results[:max_results]

    async def retrieve(
        self,
        terms: Union[str, List[str]],
        path: Union[str, Path, List[str], List[Path], None] = None,
        logic: Literal["and", "or", "not"] = "or",
        *,
        case_sensitive: bool = False,
        whole_word: bool = False,
        literal: bool = False,
        regex: bool = True,
        max_depth: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        file_type: Optional[str] = None,
        invert_match: bool = False,
        count_only: bool = False,
        line_number: bool = True,
        with_filename: bool = True,
        rank: bool = True,
        rank_kwargs: Optional[Dict] = None,
        rga_no_cache: bool = False,
        rga_cache_max_blob_len: int = 10000000,
        rga_cache_path: Optional[Union[str, Path]] = None,
        timeout: float = GREP_TIMEOUT,
    ) -> List[Dict[str, Any]]:
        """Search for terms in files using ripgrep-all, supporting AND/OR/NOT logic.

        Args:
            terms: Single pattern (str) or list of patterns (List[str]).
            path: Single path (str/Path) or multiple paths (List[str]/List[Path]) to search in (defaults to current directory).
            logic:
                - "or" (default): match any term (A OR B OR C)
                - "and": match all terms in same file (A AND B AND C)
                - "not": match first term but NOT any of the rest (A AND NOT B AND NOT C)
            case_sensitive: If True, enable case-sensitive search (`-s`).
            whole_word: If True, match whole words only (`-w`).
            literal: If True, treat patterns as literal strings (`-F`). Applies to all terms.
            regex: If False, implies `literal=True`.
            max_depth: Maximum directory depth to search (`--max-depth`).
            include: List of glob patterns to include (`-g`).
            exclude: List of glob patterns to exclude (`-g '!...'`).
            file_type: Search only files of given type (`-t <type>`), e.g., 'py', 'md'.
            invert_match: Invert match (`-v`). Note: conflicts with `logic="not"`; ignored in that case.
            count_only: Only output match counts per file (`-c`).
            line_number: Show line numbers (`-n`, default True).
            with_filename: Show filenames (`-H`, default True).
            rank: If True, rerank results by relevance score.
            rank_kwargs: Additional kwargs for ranking (see `_rerank_results`).
            rga_no_cache: If True, disable rga caching (`--rga-no-cache`).
            rga_cache_max_blob_len: Max blob length for rga cache (`--rga-cache-max-blob-len`). Defaults to 10MB.
            rga_cache_path: Custom path for rga cache (`--rga-cache-path`).
                If None, then set the path to `/path/to/your_work_path/.cache/rga`
            timeout: Maximum time in seconds to wait for the search to complete.

        Returns:
            List of match objects (from `rga --json`), or list of {'path': str, 'count': int} if `count_only=True`.
            For "and"/"not", matches correspond to the **last term** (or first, if only one) in qualifying files/lines.
        """
        results: List[Dict[str, Any]] = []

        # Normalize terms
        if isinstance(terms, str):
            terms = [terms]
        if not terms:
            return results

        rga_cache_path = rga_cache_path or self.rga_cache
        rga_cache_path = str(Path(rga_cache_path).resolve())

        # Multi-term logic routing
        if logic == "or":
            results, retrieve_pattern = await self._retrieve_or(
                terms=terms,
                path=path,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                literal=literal,
                regex=regex,
                max_depth=max_depth,
                include=include,
                exclude=exclude,
                file_type=file_type,
                invert_match=invert_match,
                count_only=count_only,
                line_number=line_number,
                with_filename=with_filename,
                rga_no_cache=rga_no_cache,
                rga_cache_max_blob_len=rga_cache_max_blob_len,
                rga_cache_path=rga_cache_path,
                timeout=timeout,
            )
        elif logic == "and":
            results = await self._retrieve_and(
                terms=terms,
                path=path,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                literal=literal,
                regex=regex,
                max_depth=max_depth,
                include=include,
                exclude=exclude,
                file_type=file_type,
                count_only=count_only,
                line_number=line_number,
                with_filename=with_filename,
                match_same_line=False,  # file-level AND (most useful)
                rga_no_cache=rga_no_cache,
                rga_cache_max_blob_len=rga_cache_max_blob_len,
                rga_cache_path=rga_cache_path,
                timeout=timeout,
            )
        elif logic == "not":
            if len(terms) < 2:
                raise ValueError(
                    "logic='not' requires at least two terms: [positive, negative1, ...]"
                )
            results = await self._retrieve_not(
                positive=terms[0],
                negatives=terms[1:],
                path=path,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                literal=literal,
                regex=regex,
                max_depth=max_depth,
                include=include,
                exclude=exclude,
                file_type=file_type,
                count_only=count_only,
                line_number=line_number,
                with_filename=with_filename,
                rga_no_cache=rga_no_cache,
                rga_cache_max_blob_len=rga_cache_max_blob_len,
                rga_cache_path=rga_cache_path,
                timeout=timeout,
            )
        else:
            raise ValueError(
                f"Unsupported logic: {logic}. Choose from 'and', 'or', 'not'."
            )

        # ====== Reranking Post-Processing ======
        if rank and not count_only and results:
            rank_kwargs = rank_kwargs or {}

            def _default_text_extractor(match: Dict) -> str:
                try:
                    return match["data"]["lines"]["text"]
                except (KeyError, TypeError):
                    return ""

            score_opts = {
                "case_sensitive": case_sensitive,
                "whole_word": whole_word,
                "length_norm": rank_kwargs.get("length_norm", "linear"),
                "base_length": rank_kwargs.get("base_length", 100),
                "exact_bonus": rank_kwargs.get("exact_bonus", 2.0),
                "tf_weight": rank_kwargs.get("tf_weight", 1.0),
                "term_weights": rank_kwargs.get("term_weights", None),
            }

            # Reconstruct file groups: [begin, match*, end]*
            grouped: List[List[Dict]] = []
            current_group: List[Dict] = []

            for item in results:
                item_type = item.get("type")
                if item_type == "begin":
                    # Start new group
                    if current_group:
                        grouped.append(current_group)
                    current_group = [item]
                elif item_type == "end":
                    # Close current group
                    current_group.append(item)
                    grouped.append(current_group)
                    current_group = []
                elif item_type == "match":
                    # Accumulate in current group
                    if current_group:  # defensive: should always be inside begin/end
                        current_group.append(item)
                    else:
                        # Orphan match? Append to new dummy group (should not happen)
                        current_group = [
                            {"type": "begin", "data": {"path": {"text": "<unknown>"}}},
                            item,
                        ]
                else:
                    # e.g., "summary" — append to current or new group
                    if current_group:
                        current_group.append(item)
                    else:
                        grouped.append([item])

            # If unclosed group remains (e.g., no final 'end'), flush it
            if current_group:
                grouped.append(current_group)

            # Process each group: rerank only the 'match' items inside
            new_results: List[Dict] = []

            for group in grouped:
                if not group:
                    continue

                # Identify begin / match / end segments
                match_items = [g for g in group if g.get("type") == "match"]

                # Rerank match items
                scored_matches = []
                for m in match_items:
                    text = _default_text_extractor(m)
                    score = self._calculate_relevance_score(
                        text=text, terms=terms, **score_opts
                    )
                    new_m = {**m, "score": score}
                    scored_matches.append((score, new_m))

                # Sort descending by score
                scored_matches.sort(key=lambda x: x[0], reverse=True)
                reranked_matches = [item for _, item in scored_matches]

                # Rebuild group in correct order:
                # [begin] + [other non-match items in original order] + [reranked matches] + [end]
                # But preserve *relative order* of non-match items (e.g., context lines)
                # Simpler: walk original group, replace match list with reranked one
                rebuilt_group = []
                match_iter = iter(reranked_matches)
                for item in group:
                    if item.get("type") == "match":
                        # Pull next from reranked list (should be same length)
                        try:
                            rebuilt_group.append(next(match_iter))
                        except StopIteration:
                            pass  # fallback: skip (should not happen)
                    else:
                        rebuilt_group.append(item)

                new_results.extend(rebuilt_group)

            results = new_results

        return results

    @staticmethod
    def _run_rga(
        args: List[str], json_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run ripgrep-all with given arguments.

        Args:
            args: List of ripgrep-all CLI arguments.
            json_output: If True, forces `--json` and parses stdout as JSON Lines.

        Returns:
            CompletedProcess object with parsed stdout (as list of dicts if json_output=True).
        """
        cmd = ["rga", "--no-config"]  # disable user config for reproducibility
        if json_output:
            cmd.append("--json")
        cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,  # we handle non-zero exit codes manually
            )

            if result.returncode != 0:
                if "ripgrep" in result.stderr.lower() or " rg " in result.stderr:
                    raise RuntimeError(
                        f"ripgrep-all depends on 'ripgrep' (rg), but it's missing: {result.stderr.strip()}"
                    )
                elif result.returncode > 1:
                    # Exit code 2: partial errors (some files failed
                    # preprocessing).  If stdout has content, continue
                    # to parse — valid matches may still be present.
                    if not result.stdout.strip():
                        raise RuntimeError(f"rga execution failed: {result.stderr.strip()}")
                    logger.warning(
                        f"rga returned exit code {result.returncode} with partial errors "
                        f"(first 300 chars): {result.stderr.strip()[:300]}"
                    )

            # Parse JSON Lines if requested (including exit code 2)
            if json_output and result.returncode in (0, 1, 2) and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                result.stdout = [json.loads(line) for line in lines if line]
            return result
        except FileNotFoundError:
            raise RuntimeError(
                "ripgrep-all ('rga') not found. Please install ripgrep-all first."
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse ripgrep-all JSON output: {e}\nRaw output: {result.stdout}"
            )

    @staticmethod
    async def _terminate_process(
        process: asyncio.subprocess.Process,
        communicate_task: Optional[asyncio.Task] = None,
    ) -> None:
        """Terminate a search process tree and wait until it is reaped."""
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif os.name == "nt" and process.returncode is None:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(
                    killer.wait(), timeout=GREP_PROCESS_KILL_TIMEOUT
                )
            elif process.returncode is None:
                process.kill()
        except (
            asyncio.TimeoutError,
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
        ):
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        try:
            if communicate_task is not None:
                await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=GREP_PROCESS_KILL_TIMEOUT,
                )
            else:
                await asyncio.wait_for(
                    process.wait(), timeout=GREP_PROCESS_KILL_TIMEOUT
                )
        except (asyncio.TimeoutError, ProcessLookupError):
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=GREP_PROCESS_KILL_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Search process {} could not be reaped within {:.1f}s",
                        process.pid,
                        GREP_PROCESS_KILL_TIMEOUT,
                    )

    @staticmethod
    def _parse_async_result(
        command: str,
        returncode: int,
        stdout: bytes,
        stderr: bytes,
        *,
        json_output: bool,
    ) -> Dict[str, Any]:
        """Decode and validate output from an asynchronous search command."""
        stdout_str = stdout.decode(errors="replace").strip()
        stderr_str = stderr.decode(errors="replace").strip()

        if returncode != 0 and returncode > 1:
            if not stdout_str:
                raise RuntimeError(
                    f"{command} execution failed with code {returncode}: {stderr_str}"
                )
            logger.warning(
                f"{command} returned exit code {returncode} with partial errors "
                f"(first 300 chars): {stderr_str[:300]}"
            )

        parsed_stdout: Union[str, List[Dict[str, Any]]] = stdout_str
        if json_output and returncode in (0, 1, 2) and stdout_str:
            try:
                parsed_stdout = [
                    json.loads(line) for line in stdout_str.splitlines() if line
                ]
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse {command} JSON output: {exc}"
                ) from exc

        return {
            "returncode": returncode,
            "stdout": parsed_stdout,
            "stderr": stderr_str,
            "search_backend": command,
        }

    @staticmethod
    async def _run_search_process(
        command: str,
        args: List[str],
        *,
        semaphore: asyncio.Semaphore,
        queue_timeout: float,
        execution_timeout: float,
        json_output: bool,
    ) -> Dict[str, Any]:
        """Run a bounded search process with cancellation-safe cleanup."""
        queued_at = time.perf_counter()
        acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
            acquired = True
            queue_wait_ms = (time.perf_counter() - queued_at) * 1000
        except asyncio.TimeoutError as exc:
            raise SearchProcessTimeoutError(
                command, "queue wait", queue_timeout
            ) from exc

        process: Optional[asyncio.subprocess.Process] = None
        try:
            cmd = [command, "--no-config"]
            if json_output:
                cmd.append("--json")
            cmd.extend(args)

            process_kwargs: Dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if os.name == "posix":
                process_kwargs["start_new_session"] = True
            elif os.name == "nt":
                process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            execution_started = time.perf_counter()
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd, **process_kwargs
                )
            except FileNotFoundError as exc:
                raise SearchExecutableNotFoundError(command) from exc

            communicate_task = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communicate_task), timeout=execution_timeout
                )
            except asyncio.TimeoutError as exc:
                await asyncio.shield(
                    GrepRetriever._terminate_process(process, communicate_task)
                )
                raise SearchProcessTimeoutError(
                    command, "execution", execution_timeout
                ) from exc
            except asyncio.CancelledError:
                await asyncio.shield(
                    GrepRetriever._terminate_process(process, communicate_task)
                )
                raise

            result = GrepRetriever._parse_async_result(
                command,
                process.returncode,
                stdout,
                stderr,
                json_output=json_output,
            )
            result["queue_wait_ms"] = queue_wait_ms
            result["execution_ms"] = (
                time.perf_counter() - execution_started
            ) * 1000
            return result
        finally:
            if acquired:
                semaphore.release()

    @staticmethod
    def _rg_compatible_args(args: List[str]) -> List[str]:
        """Remove rga-only flags before falling back to native ripgrep."""
        return [arg for arg in args if not arg.startswith("--rga-")]

    @staticmethod
    async def _run_rga_async(
        args: List[str],
        json_output: bool = True,
        timeout: float = GREP_TIMEOUT,
        allow_rg_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Run rga, falling back to rg when rga is unavailable or times out."""
        global _RGA_BACKOFF_UNTIL
        started_at = time.perf_counter()

        if (
            allow_rg_fallback
            and GREP_FALLBACK_TO_RG
            and time.monotonic() < _RGA_BACKOFF_UNTIL
        ):
            result = await GrepRetriever._run_search_process(
                "rg",
                GrepRetriever._rg_compatible_args(args),
                semaphore=RG_FALLBACK_SEMAPHORE,
                queue_timeout=GREP_QUEUE_TIMEOUT,
                execution_timeout=GREP_FALLBACK_TIMEOUT,
                json_output=json_output,
            )
            result["fallback_reason"] = "rga circuit breaker is open"
            result["fallback_used"] = True
            result["fallback_elapsed_ms"] = result.get("execution_ms", 0.0)
            result["total_elapsed_ms"] = (time.perf_counter() - started_at) * 1000
            return result

        try:
            result = await GrepRetriever._run_search_process(
                "rga",
                args,
                semaphore=RGA_SEMAPHORE,
                queue_timeout=GREP_QUEUE_TIMEOUT,
                execution_timeout=timeout,
                json_output=json_output,
            )
            result["fallback_used"] = False
            result["fallback_elapsed_ms"] = 0.0
            result["total_elapsed_ms"] = (time.perf_counter() - started_at) * 1000
            return result
        except (
            SearchProcessTimeoutError,
            SearchExecutableNotFoundError,
            RuntimeError,
        ) as exc:
            if not GREP_FALLBACK_TO_RG or not allow_rg_fallback:
                raise
            if (
                isinstance(exc, SearchExecutableNotFoundError)
                or (
                    isinstance(exc, SearchProcessTimeoutError)
                    and exc.stage == "execution"
                )
            ):
                _RGA_BACKOFF_UNTIL = max(
                    _RGA_BACKOFF_UNTIL,
                    time.monotonic() + GREP_RGA_BACKOFF_SECONDS,
                )
            logger.warning("{} Falling back to native rg.", exc)
            fallback_started = time.perf_counter()
            result = await GrepRetriever._run_search_process(
                "rg",
                GrepRetriever._rg_compatible_args(args),
                semaphore=RG_FALLBACK_SEMAPHORE,
                queue_timeout=GREP_QUEUE_TIMEOUT,
                execution_timeout=GREP_FALLBACK_TIMEOUT,
                json_output=json_output,
            )
            result["fallback_reason"] = str(exc)
            result["fallback_used"] = True
            result["fallback_elapsed_ms"] = (
                time.perf_counter() - fallback_started
            ) * 1000
            result["primary_failure_stage"] = getattr(exc, "stage", "execution")
            result["total_elapsed_ms"] = (time.perf_counter() - started_at) * 1000
            return result

    @staticmethod
    async def _retrieve_single(**kwargs) -> List[Dict[str, Any]]:
        """Wrapper for original single-pattern search (extracted for reuse)."""
        pattern = kwargs.pop("pattern")
        args = []

        # Basic ripgrep-all args
        regex = kwargs.get("regex", True)
        literal = kwargs.get("literal", False)
        case_sensitive = kwargs.get("case_sensitive", False)
        whole_word = kwargs.get("whole_word", False)
        invert_match = kwargs.get("invert_match", False)
        count_only = kwargs.get("count_only", False)
        line_number = kwargs.get("line_number", True)
        with_filename = kwargs.get("with_filename", True)
        max_depth = kwargs.get("max_depth")
        include = kwargs.get("include")
        exclude = kwargs.get("exclude")
        file_type = kwargs.get("file_type")
        path = kwargs.get("path")
        timeout = kwargs.get("timeout", GREP_TIMEOUT)

        # Additional ripgrep-all args
        rga_no_cache = kwargs.get("rga_no_cache", False)
        rga_cache_max_blob_len = kwargs.get(
            "rga_cache_max_blob_len", 10000000
        )  # Default 10MB
        rga_cache_path = kwargs.get("rga_cache_path")

        # Build argument list
        if not regex:
            literal = True
        if literal:
            args.append("-F")
        if case_sensitive:
            args.append("-s")
        else:
            args.append("-i")
        if whole_word:
            args.append("-w")
        if invert_match:
            args.append("-v")
        if count_only:
            args.append("-c")
        if not line_number:
            args.append("--no-line-number")
        if not with_filename:
            args.append("--no-filename")
        if max_depth is not None:
            args.extend(["--max-depth", str(max_depth)])
        if include:
            for inc in include:
                args.extend(["-g", inc])
        if exclude:
            for exc in exclude:
                args.extend(["-g", f"!{exc}"])
        if file_type:
            args.extend(["-t", file_type])

        if rga_no_cache:
            args.append("--rga-no-cache")

        args.extend([f"--rga-cache-max-blob-len={str(rga_cache_max_blob_len)}"])

        if rga_cache_path:
            args.extend([f"--rga-cache-path={str(rga_cache_path)}"])

        # Corpus-agnostic cost bounds for the query hot path (see AGENTS.md §9.5):
        #   * --max-filesize skips any oversized blob regardless of type.
        #   * --rga-adapters keeps only bounded document extractors and drops the
        #     unbounded recursive/streaming adapters, so archives are never
        #     inline-decompressed during a query.
        # --max-filesize is shared with ripgrep and survives the rg fallback;
        # --rga-adapters is rga-only and is stripped by _rg_compatible_args.
        if GREP_MAX_FILESIZE_MB > 0:
            args.append(f"--max-filesize={GREP_MAX_FILESIZE_MB}M")
        if GREP_RGA_ADAPTERS:
            args.append(f"--rga-adapters={GREP_RGA_ADAPTERS}")

        # Resolve the search paths into a suffix reused by both tiered passes.
        path_args: List[str] = []
        if path is not None:
            if isinstance(path, (str, Path)):
                path_args.append(str(path))
            elif isinstance(path, list):
                path_args.extend(str(p) for p in path)
            else:
                raise TypeError(f"Unsupported type for 'path': {type(path)}")

        # P1 tiered scan applies only to a broad content search. When the caller
        # constrains the file set (include globs / ripgrep --type) or wants only
        # counts, honour that intent with the original single rga pass.
        tiered = (
            GREP_TIERED_SCAN
            and not count_only
            and not include
            and not file_type
            and bool(GREP_RICH_EXTENSIONS)
        )

        started_at = time.perf_counter()
        if not tiered:
            result = await GrepRetriever._run_rga_async(
                args=args + [pattern] + path_args,
                json_output=not count_only,
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            return GrepRetriever._parse_single_result(
                result,
                count_only=count_only,
                rga_no_cache=rga_no_cache,
                elapsed_ms=elapsed_ms,
            )

        # Text pass: native rg over everything. rg auto-detects and skips binary
        # files and never decompresses archives, so this is O(bytes) fast and
        # covers all plain-text/markup/code formats (including extensionless).
        text_args = GrepRetriever._rg_compatible_args(list(args))
        # Cap matches per file on the text pass ONLY: rg over the whole tree can
        # match a huge number of files (a broad term across a large text corpus),
        # and unbounded --json output there is what blows the scan budget. The
        # rga rich pass stays uncapped so a needle deep inside a single pdf/docx
        # is never truncated out of the evidence.
        if GREP_MAX_MATCHES_PER_FILE > 0:
            text_args.append(f"--max-count={GREP_MAX_MATCHES_PER_FILE}")
        text_args = text_args + [pattern] + path_args
        # Rich pass: rga restricted to the binary document formats rg cannot read.
        rich_globs: List[str] = []
        for ext in GREP_RICH_EXTENSIONS:
            rich_globs.extend(["-g", f"*.{ext}"])
        rich_args = args + rich_globs + [pattern] + path_args

        text_result, rich_result = await asyncio.gather(
            GrepRetriever._run_search_process(
                "rg",
                text_args,
                semaphore=RG_FALLBACK_SEMAPHORE,
                queue_timeout=GREP_QUEUE_TIMEOUT,
                execution_timeout=GREP_TEXT_TIMEOUT,
                json_output=True,
            ),
            GrepRetriever._run_rga_async(
                args=rich_args, json_output=True, timeout=timeout,
            ),
            return_exceptions=True,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        merged: List[Dict[str, Any]] = []
        for label, res in (("rg", text_result), ("rga", rich_result)):
            if isinstance(res, Exception):
                logger.warning("tiered {} pass failed: {}", label, res)
                continue
            try:
                merged.extend(
                    GrepRetriever._parse_single_result(
                        res,
                        count_only=False,
                        rga_no_cache=rga_no_cache,
                        elapsed_ms=elapsed_ms,
                    )
                )
            except Exception as exc:  # a genuine backend failure in one pass
                logger.warning("tiered {} parse failed: {}", label, exc)
        return merged

    @staticmethod
    def _parse_single_result(
        result: Dict[str, Any],
        *,
        count_only: bool,
        rga_no_cache: bool,
        elapsed_ms: float,
    ) -> List[Dict[str, Any]]:
        """Parse one rga/rg subprocess result into match dicts (or count dicts).

        Exit codes: 0 = matches found, 1 = no matches, 2 = partial errors (some
        files failed preprocessing but results may still be present).
        """
        returncode = result["returncode"]
        stderr_str = result.get("stderr", "").strip()
        if returncode in (0, 2):
            if count_only:
                counts: List[Dict[str, Any]] = []
                raw = result["stdout"]
                if isinstance(raw, str):
                    for line in raw.strip().splitlines():
                        if ":" in line:
                            p, c = line.rsplit(":", 1)
                            try:
                                counts.append({"path": p, "count": int(c)})
                            except ValueError:
                                continue
                return counts
            stdout = result["stdout"]
            parsed = stdout if isinstance(stdout, list) else []
            backend = str(result.get("search_backend", "rga"))
            fallback_reason = result.get("fallback_reason")
            for item in parsed:
                if isinstance(item, dict):
                    item.setdefault("_search_backend", backend)
                    item.setdefault("_search_elapsed_ms", elapsed_ms)
                    item.setdefault(
                        "_search_queue_wait_ms", result.get("queue_wait_ms", 0.0),
                    )
                    item.setdefault(
                        "_search_execution_ms", result.get("execution_ms", 0.0),
                    )
                    item.setdefault(
                        "_fallback_elapsed_ms", result.get("fallback_elapsed_ms", 0.0),
                    )
                    item.setdefault("_search_cache_enabled", not rga_no_cache)
                    if result.get("primary_failure_stage"):
                        item.setdefault(
                            "_primary_failure_stage", result["primary_failure_stage"],
                        )
                    if fallback_reason:
                        item.setdefault("_fallback_reason", fallback_reason)
            if returncode == 2 and not parsed and stderr_str:
                logger.warning(
                    f"rga exit 2 with no results — preprocessing may have failed "
                    f"(missing poppler-utils/pandoc?): {stderr_str[:300]}"
                )
            return parsed
        elif returncode == 1:
            return []
        else:
            raise RuntimeError(
                f"ripgrep-all failed (exit {returncode}): {stderr_str}"
            )

    async def prewarm_rich_cache(
        self,
        path: Union[str, Path, List[str], List[Path]],
        *,
        max_depth: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Populate the rga cache for binary document formats ahead of queries.

        Runs the rga rich pass with a token that essentially never occurs in
        prose, so poppler/pandoc conversions are cached without producing output.
        This amortizes the cold rich-format extraction cost (which the tiered
        scan otherwise pays on the first query over pdf/docx-heavy corpora) into
        an offline / compile step. Best effort: failures are swallowed so they
        never block the caller.
        """
        if not GREP_RICH_EXTENSIONS:
            return {"prewarmed": False, "reason": "no rich extensions configured"}
        paths = (
            [str(path)] if isinstance(path, (str, Path))
            else [str(p) for p in path]
        )
        rga_cache_path = str(Path(self.rga_cache).resolve())
        args: List[str] = [
            "-i",
            "--rga-cache-max-blob-len=10000000",
            f"--rga-cache-path={rga_cache_path}",
        ]
        if GREP_MAX_FILESIZE_MB > 0:
            args.append(f"--max-filesize={GREP_MAX_FILESIZE_MB}M")
        if GREP_RGA_ADAPTERS:
            args.append(f"--rga-adapters={GREP_RGA_ADAPTERS}")
        if max_depth is not None:
            args.extend(["--max-depth", str(max_depth)])
        for ext in GREP_RICH_EXTENSIONS:
            args.extend(["-g", f"*.{ext}"])
        args.append("zzqx_prewarm_nomatch_xqzz")
        args.extend(paths)
        started = time.perf_counter()
        try:
            await GrepRetriever._run_rga_async(
                args=args,
                json_output=False,
                timeout=timeout or GREP_TIMEOUT,
                allow_rg_fallback=False,
            )
            return {
                "prewarmed": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as exc:
            logger.warning("rich cache prewarm failed: {}", exc)
            return {"prewarmed": False, "reason": str(exc)}

    @staticmethod
    async def _retrieve_or(
        terms: List[str],
        **kwargs,
    ) -> (List[Dict[str, Any]], str):
        """OR: Match any term.

        When ``literal=True`` (``-F`` mode), ripgrep treats the *entire*
        pattern as a fixed string — ``|`` is NOT interpreted as alternation.
        In that case we must search each term individually and merge.
        When ``literal=False`` (regex mode), we use ``|`` alternation
        normally.
        """
        import asyncio as _aio

        literal = kwargs.get("literal", False)
        if literal:
            if len(terms) == 1:
                # Single term — safe to use -F directly
                result = await GrepRetriever._retrieve_single(
                    pattern=terms[0], **kwargs
                )
                return result, terms[0]

            # Multiple terms + literal mode: search each term separately
            # then merge results to simulate OR. Bound fan-out and retain
            # successful terms when another backend attempt fails.
            term_semaphore = _aio.Semaphore(GREP_KEYWORD_CONCURRENT_LIMIT)

            async def _search_one(term: str):
                async with term_semaphore:
                    return await GrepRetriever._retrieve_single(
                        pattern=term, **kwargs
                    )

            results_lists = await _aio.gather(
                *[_search_one(term) for term in terms],
                return_exceptions=True,
            )

            combined: List[Dict[str, Any]] = []
            failures: List[Exception] = []
            for result in results_lists:
                if isinstance(result, _aio.CancelledError):
                    raise result
                if isinstance(result, Exception):
                    failures.append(result)
                    continue
                combined.extend(result)
            if not combined and failures:
                raise failures[0]

            pattern_desc = " | ".join(terms)
            return combined, pattern_desc
        else:
            # Wrap each term in (?:...) to avoid precedence issues
            pattern = "|".join(f"(?:{term})" for term in terms)
            kwargs["literal"] = False

            result = await GrepRetriever._retrieve_single(pattern=pattern, **kwargs)

            return result, pattern

    @staticmethod
    async def _retrieve_and(
        terms: List[str],
        match_same_line: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """AND: All terms in same file (default) or same line."""
        count_only = kwargs.get("count_only", False)

        # Step 1: Get files containing first term
        first_matches = await GrepRetriever._retrieve_single(pattern=terms[0], **kwargs)
        if not first_matches:
            return []

        if match_same_line:
            # Line-level AND: filter lines containing all terms
            def line_contains_all(
                line: str, others: List[str], case_sensitive: bool
            ) -> bool:
                if not case_sensitive:
                    line = line.lower()
                    others = [t.lower() for t in others]
                return all(term in line for term in others)

            case_sensitive = kwargs.get("case_sensitive", False)
            others = terms[1:]
            return [
                m
                for m in first_matches
                if m["type"] == "match"
                and line_contains_all(
                    m["data"]["lines"]["text"], others, case_sensitive
                )
            ]
        else:
            # File-level AND (default)
            files_with_first = {
                m["data"]["path"]["text"] for m in first_matches if m["type"] == "match"
            }
            qualifying_files = set()

            for f in files_with_first:
                valid = True
                for term in terms[1:]:
                    res = await GrepRetriever._retrieve_single(
                        pattern=term,
                        path=f,
                        count_only=count_only,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ["path", "count_only"]
                        },
                    )
                    if not res:  # no match
                        valid = False
                        break
                if valid:
                    qualifying_files.add(f)

            # Collect matches for last term (or first) in qualifying files
            target_term = terms[-1]
            all_matches = []
            kwargs["pattern"] = target_term
            for f in qualifying_files:
                kwargs["path"] = f
                matches = await GrepRetriever._retrieve_single(**kwargs)
                all_matches.extend(matches)
            return all_matches

    @staticmethod
    async def _retrieve_not(
        positive: str,
        negatives: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """NOT: Match positive term, but exclude files/lines containing any negative term."""
        # count_only = kwargs.get("count_only", False)

        # Step 1: Get matches for positive term
        pos_matches = await GrepRetriever._retrieve_single(pattern=positive, **kwargs)
        if not pos_matches:
            return []

        # Decide: file-level NOT (default) vs line-level NOT
        # We use file-level for efficiency (avoid per-line Python filtering on large outputs)
        files_with_positive = {
            m["data"]["path"]["text"] for m in pos_matches if m["type"] == "match"
        }
        excluded_files = set()

        for f in files_with_positive:
            for neg in negatives:
                res = await GrepRetriever._retrieve_single(
                    pattern=neg,
                    path=f,
                    count_only=True,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["path", "count_only"]
                    },
                )
                if res:  # found negative → exclude this file
                    excluded_files.add(f)
                    break

        # Keep only matches from non-excluded files
        kept_matches = [
            m
            for m in pos_matches
            if m["type"] == "match" and m["data"]["path"]["text"] not in excluded_files
        ]

        return kept_matches

    async def list_files(
        self,
        path: Optional[str] = None,
        *,
        max_depth: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        file_type: Optional[str] = None,
        hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> List[str]:
        """List files that would be searched by ripgrep-all (like `rga --files`).

        Args:
            path: Path to list files in.
            max_depth: Maximum directory depth.
            include: Glob patterns to include.
            exclude: Glob patterns to exclude.
            file_type: Restrict to file type (e.g., 'py').
            hidden: Include hidden files/dirs (`--hidden`).
            follow_symlinks: Follow symbolic links (`--follow`).

        Returns:
            List of relative file paths (strings).
        """
        args = ["--files"]
        if max_depth is not None:
            args.extend(["--max-depth", str(max_depth)])
        if include:
            for inc in include:
                args.extend(["-g", inc])
        if exclude:
            for exc in exclude:
                args.extend(["-g", f"!{exc}"])
        if file_type:
            args.extend(["-t", file_type])
        if hidden:
            args.append("--hidden")
        if follow_symlinks:
            args.append("--follow")
        if path:
            args.append(path)

        result: Dict[str, Any] = await GrepRetriever._run_rga_async(
            args, json_output=False
        )
        if result["returncode"] not in (0, 1):
            raise RuntimeError(
                f"ripgrep-all --files failed: {result['stderr'].strip()}"
            )

        return result["stdout"].strip().splitlines() if result["stdout"].strip() else []

    async def retrieve_by_filename(
        self,
        patterns: Union[str, List[str]],
        path: Union[str, Path, List[str], List[Path], None] = None,
        *,
        case_sensitive: bool = False,
        max_depth: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        file_type: Optional[str] = None,
        rank: bool = True,
        timeout: float = GREP_TIMEOUT,
    ) -> List[Dict[str, Any]]:
        """Search for files by filename patterns (fast file name matching).
        
        This method performs filename-only search without reading file contents,
        making it significantly faster than content-based search.
        
        Args:
            patterns: Single pattern (str) or list of patterns (List[str]) to match filenames.
                     Patterns are treated as regex by default (e.g., "test.*\\.py").
            path: Single path (str/Path) or multiple paths (List[str]/List[Path]) to search in.
            case_sensitive: If True, enable case-sensitive filename matching.
            max_depth: Maximum directory depth to search.
            include: List of glob patterns to include (e.g., ["*.py", "*.md"]).
            exclude: List of glob patterns to exclude (e.g., ["*.pyc", "*.log"]).
            file_type: Search only files of given type (e.g., 'py', 'md').
            rank: If True, rank results by pattern match quality (e.g., exact match > partial match).
            timeout: Maximum time in seconds to wait for the search to complete.
        
        Returns:
            List of match objects with structure:
            [
                {
                    'path': '/absolute/path/to/file.py',
                    'filename': 'file.py',
                    'match_score': 1.0,  # relevance score (0.0-1.0)
                    'type': 'filename_match'
                },
                ...
            ]
        """
        # Normalize patterns
        if isinstance(patterns, str):
            patterns = [patterns]
        
        logger.debug(f"retrieve_by_filename called with patterns: {patterns}, path: {path}, "
                    f"include: {include}, exclude: {exclude}, max_depth: {max_depth}")
        
        # Normalize paths
        if path is None:
            paths = ["."]
        elif isinstance(path, (str, Path)):
            paths = [str(path)]
        else:
            paths = [str(p) for p in path]
        
        # List all files in the specified paths
        all_files = []
        for search_path in paths:
            try:
                files = await self.list_files(
                    path=search_path,
                    max_depth=max_depth,
                    include=include,
                    exclude=exclude,
                    file_type=file_type,
                )
                all_files.extend(files)
            except Exception as e:
                logger.warning(f"Failed to list files in {search_path}: {e}")
                continue
        
        if not all_files:
            logger.debug("No files found to search")
            return []
        
        logger.debug(f"Searching through {len(all_files)} files with patterns: {patterns}")
        
        # Filter files by patterns
        results = []
        for file_path in all_files:
            # Get both absolute and relative paths for proper handling
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            
            # Check if filename matches any pattern
            for pattern in patterns:
                try:
                    # Compile regex pattern
                    flags = 0 if case_sensitive else re.IGNORECASE
                    regex = re.compile(pattern, flags)
                    
                    match = regex.search(filename)
                    if match:
                        logger.debug(f"Pattern '{pattern}' matched file: {filename}")
                        
                        # Calculate match score
                        match_score = self._calculate_filename_match_score(
                            filename=filename,
                            pattern=pattern,
                            case_sensitive=case_sensitive
                        )
                        
                        # Use absolute path if file exists, otherwise keep original path
                        try:
                            abs_path = str(file_path_obj.resolve())
                        except (OSError, RuntimeError):
                            abs_path = str(file_path_obj.absolute()) if file_path_obj.is_absolute() else file_path
                        
                        results.append({
                            'path': abs_path,
                            'filename': filename,
                            'match_score': match_score,
                            'type': 'filename_match',
                            'matched_pattern': pattern,
                        })
                        break  # Only count each file once (first matching pattern)
                
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                    continue
        
        logger.debug(f"Found {len(results)} matching files")
        
        # Rank results by match score if requested
        if rank and results:
            results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results

    @staticmethod
    def _calculate_filename_match_score(
        filename: str,
        pattern: str,
        case_sensitive: bool = False
    ) -> float:
        """Calculate relevance score for filename pattern match.
        
        Args:
            filename: The filename that matched
            pattern: The regex pattern that was matched
            case_sensitive: Whether the match was case-sensitive
        
        Returns:
            Score between 0.0 and 1.0, where:
            - 1.0 = exact match (highest priority)
            - 0.9 = exact match with different case
            - 0.7-0.8 = starts with pattern
            - 0.5-0.6 = contains pattern
            - 0.3-0.4 = partial regex match
        """
        # Normalize for comparison
        fn_lower = filename.lower()
        pattern_lower = pattern.lower()
        
        # Remove regex special characters for literal comparison
        pattern_literal = re.sub(r'[.*+?^${}()|[\]\\]', '', pattern)
        pattern_literal_lower = pattern_literal.lower()
        
        # Exact match (case-sensitive)
        if filename == pattern or filename == pattern_literal:
            return 1.0
        
        # Exact match (case-insensitive)
        if not case_sensitive and (fn_lower == pattern_lower or fn_lower == pattern_literal_lower):
            return 0.9
        
        # Starts with pattern
        if filename.startswith(pattern_literal):
            return 0.8
        if fn_lower.startswith(pattern_literal_lower):
            return 0.75
        
        # Contains pattern (full)
        if pattern_literal in filename:
            return 0.6
        if pattern_literal_lower in fn_lower:
            return 0.55
        
        # Partial match (proportional to match length)
        match_ratio = len(pattern_literal) / max(len(filename), 1)
        return 0.3 + (match_ratio * 0.2)  # Score between 0.3 and 0.5

    def file_types(self) -> Dict[str, List[str]]:
        """List supported file types and their associated globs/extensions.

        Returns:
            Dict mapping type names (e.g., 'python') to list of globs (e.g., ['*.py', '*.pyi']).
        """
        result = subprocess.run(
            ["rga", "--type-list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        types = {}
        for line in result.stdout.strip().splitlines():
            if ":" in line:
                name, globs = line.split(":", 1)
                types[name.strip()] = [g.strip() for g in globs.split(",") if g.strip()]
        return types

    async def replace(
        self,
        pattern: str,
        replacement: str,
        path: Optional[str] = None,
        *,
        dry_run: bool = False,
        case_sensitive: bool = False,
        literal: bool = False,
        whole_word: bool = False,
        max_depth: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform search-and-replace using ripgrep-all (via `--replace`).

        Caution: This modifies files in-place if dry_run=False.

        Args:
            pattern: Regex pattern to search for.
            replacement: Replacement string (supports $1, $2, etc.).
            path: Path to operate on.
            dry_run: If True, only show matches/replacements, don't modify files.
            case_sensitive: Enable case-sensitive matching.
            literal: Treat pattern as literal string.
            whole_word: Match whole words only.
            max_depth: Max search depth.
            include/exclude: Globs to include/exclude.

        Returns:
            List of replacement events (from `rga --json` output).
        """
        args = ["--replace", replacement]
        if dry_run:
            args.append("--dry-run")
        else:
            args.append("--passthru")  # needed for in-place replace with --replace
        if case_sensitive:
            args.append("-s")
        else:
            args.append("-i")
        if literal:
            args.append("-F")
        if whole_word:
            args.append("-w")
        if max_depth is not None:
            args.extend(["--max-depth", str(max_depth)])
        if include:
            for inc in include:
                args.extend(["-g", inc])
        if exclude:
            for exc in exclude:
                args.extend(["-g", f"!{exc}"])

        args.append(pattern)
        if path:
            args.append(path)

        result = await GrepRetriever._run_rga_async(
            args,
            allow_rg_fallback=False,
        )
        if result["returncode"] not in (0, 1):
            raise RuntimeError(
                f"ripgrep-all replace failed: {result['stderr'].strip()}"
            )

        return result["stdout"]

    def version(self) -> str:
        """Get ripgrep-all version string.

        Returns:
            Version string.
        """
        result = subprocess.run(
            ["rga", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")[0]

    def supports_feature(self, feature: str) -> bool:
        """
        Check if ripgrep-all supports a given feature (e.g., 'pcre2', 'json').

        Args:
            feature: Feature name (e.g., 'json', 'pcre2', 'lz4').

        Returns:
            True if feature is available in this rga build.
        """
        result = subprocess.run(
            ["rga", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return f"--{feature}" in result.stdout

    @staticmethod
    def _calculate_relevance_score(
        text: str,
        terms: List[str],
        *,
        case_sensitive: bool = False,
        whole_word: bool = False,
        length_norm: Literal["linear", "log", "none"] = "linear",
        base_length: int = 100,
        exact_bonus: float = 2.0,
        tf_weight: float = 1.0,
        term_weights: Optional[List[float]] = None,
        tf_saturation: Literal["log", "sigmoid", "none"] = "sigmoid",
        saturation_k: float = 1.0,
        idf_simulate: bool = True,
    ) -> float:
        """
        Compute a relevance score for a text w.r.t. a list of query terms.

        Scoring formula (per term, then summed):
            term_score = (TF_term ** tf_weight) * bonus_term * length_factor

        Where:
            - TF_term = number of matches for the term
            - bonus_term = exact_bonus if at least one match is *isolated*, else 1.0
            - length_factor = global penalty for long texts (shared across terms)

        An *isolated* match means the term is surrounded by non-alphanumeric characters
        or string boundaries (i.e., standalone token — higher relevance).

        Args:
            text: Text to score (e.g., matching line or surrounding context).
            terms: List of query terms (e.g., ["TODO", "fix"]).
            case_sensitive: Whether matching is case-sensitive.
            whole_word: If True, only matches bounded by non-word chars are counted.
            length_norm: Length penalty strategy ('linear' / 'log' / 'none').
            base_length: Scaling for linear norm (default 100 chars).
            exact_bonus: Bonus multiplier for isolated matches (e.g., 2.0).
            tf_weight: Exponent for term frequency (e.g., 0.5 for sqrt(TF)).
            term_weights: Optional weights for each term (default: uniform 1.0).
                         Must have same length as `terms` if provided.
            tf_saturation: How to saturate term frequency:
                - 'log':    tf_adj = 1 + log(tf)               (smooth, BM25-like)
                - 'sigmoid':tf_adj = tf / (tf + k)             (bounded [0,1))
                - 'none':   tf_adj = tf                         (original)
            saturation_k: Parameter for sigmoid (default 1.0); larger → slower saturation.
            idf_simulate: If True, penalize short/common terms heuristically:
                idf_factor = max(0.1, min(1.0, len(term) / 5))   # e.g., "a" → 0.2, "error" → 1.0

        Returns:
            Non-negative float relevance score (sum of term scores).
        """
        if not text or not terms:
            return 0.0

        if term_weights is not None:
            if len(term_weights) != len(terms):
                raise ValueError("term_weights must have same length as terms")
        else:
            term_weights = [1.0] * len(terms)

        flags = 0 if case_sensitive else re.IGNORECASE

        # Precompute length factor (shared)
        n = len(text)
        if length_norm == "linear":
            length_factor = 1.0 / (n / base_length + 1)
        elif length_norm == "log":
            length_factor = 1.0 / (math.log(n + 1) + 1)
        else:  # "none"
            length_factor = 1.0

        total_score = 0.0

        for term, weight in zip(terms, term_weights):
            if not term:
                continue

            if idf_simulate:
                # Normalize: len=1→0.2, len=5+→1.0 (clamped)
                idf_factor = max(0.2, min(1.0, len(term) / 5.0))
            else:
                idf_factor = 1.0

            escaped = re.escape(term)
            if whole_word:
                regex = rf"(?<!\w){escaped}(?!\w)"
            else:
                regex = escaped

            # Find matches
            try:
                matches_iter = re.finditer(regex, text, flags=flags)
                match_positions: List[int] = [m.start() for m in matches_iter]
                tf = len(match_positions)
            except re.error as e:
                logger.warning(f"Regex failed for term {term!r}: {e}", RuntimeWarning)
                tf = 0

            if tf == 0:
                continue

            # Isolation bonus
            has_isolated_match = any(
                (pos == 0 or not text[pos - 1].isalnum())
                and (
                    pos + len(term) == len(text) or not text[pos + len(term)].isalnum()
                )
                for pos in match_positions
            )
            bonus = exact_bonus if has_isolated_match else 1.0

            if tf_saturation == "log":
                tf_adj = 1.0 + math.log(tf)  # log(1)=0 → tf=1 → tf_adj=1.0
            elif tf_saturation == "sigmoid":
                tf_adj = tf / (tf + saturation_k)  # e.g., k=1: tf=1→0.5, tf=10→0.91
            elif tf_saturation == "none":
                tf_adj = float(tf)
            else:
                raise ValueError(f"Unknown tf_saturation: {tf_saturation}")

            # Term score with saturation + IDF + bonus + weight
            term_score = (tf_adj**tf_weight) * bonus * weight * idf_factor
            total_score += term_score

        score = total_score * length_factor
        return max(0.0, score)

    @staticmethod
    def merge_results(
        raw_results: List[Dict[str, Any]], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Merge ripgrep-all --json output into a structured per-file result list.

        This function:
          - Groups 'match' entries by file path (bounded by 'begin' and 'end' events).
          - For each file, collects 'match' items and sorts them by score (desc).
          - Takes top-`limit` matches per file (default: 50).
          - Combines all lines.text from selected matches into a list (in original match order).
          - Returns a list of unified file results.

        Args:
            raw_results: List of parsed JSON objects from `rga --json` output.
            limit: Maximum number of match items to keep per file (default: 50).

        Returns:
            A list of dictionaries, each representing one file with:
              - "path": str
              - "matches": List[Dict]  # top `limit` match items (sorted by score desc)
              - "lines": List[str]     # lines.text from those matches, in match order
              - "total_matches": int   # total matches found in this file (before limit)
        """
        if not raw_results:
            return []

        # State tracking
        current_path: Optional[str] = None
        file_matches: List[Dict[str, Any]] = []  # matches for current file
        all_files: List[Dict[str, Any]] = []  # final result accumulator

        for item in raw_results:
            item_type = item.get("type")
            data = item.get("data", {})

            if item_type == "begin":
                # Start a new file context
                path_obj = data.get("path", {})
                current_path = path_obj.get("text")
                file_matches = []  # reset match buffer

            elif item_type == "match" and current_path is not None:
                # Accumulate match; retain full item for sorting & line extraction
                # Note: 'score' is top-level in your example (not in 'data')
                file_matches.append(item)

            elif item_type == "end":
                # Finalize current file
                if current_path is not None:
                    # Sort matches by score (descending); assume score exists
                    file_matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    total_count = len(file_matches)
                    top_matches = file_matches[:limit]

                    # Extract lines.text in match order (not sorted order if stable sort not guaranteed)
                    # But since we sort, we preserve sorted order → lines follow score order
                    lines = [
                        match["data"]["lines"]["text"]
                        for match in top_matches
                        if "data" in match and "lines" in match["data"]
                    ]

                    all_files.append(
                        {
                            "path": current_path,
                            "matches": top_matches,  # full match objects
                            "lines": lines,  # list of line strings
                            "total_matches": total_count,  # before limiting
                            "total_score": 0.0,
                        }
                    )

                # Reset
                current_path = None
                file_matches = []

            # Ignore "summary" and unknown types

        return all_files


class TextRetriever(GrepRetriever):
    """Alias for GrepRetriever for backward compatibility."""

    pass
