import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger

from ..utils.file_utils import StorageStructure
from .base import BaseRetriever


class GrepRetriever(BaseRetriever):
    """A Python wrapper for ripgrep-all (rga), exposing its functionality via static methods.

    All methods are static and return parsed results. JSON output is preferred where possible
    for reliable parsing. Shell injection is mitigated by using `subprocess.run` with `shell=False`
    and explicit argument lists.

    For more information about ripgrep-all, please refer to `https://github.com/phiresky/ripgrep-all`
    """

    def __init__(self, work_path: Union[str, Path], **kwargs):
        super().__init__()

        self.work_path: Path = Path(work_path)
        self.rga_cache: Path = (
            self.work_path / StorageStructure.CACHE_DIR / StorageStructure.GREP_DIR
        )
        self.rga_cache.mkdir(parents=True, exist_ok=True)

        try:
            self.init_rga()
        except Exception as e:
            logger.warning(
                f"Failed to initialize `ripgrep-all`: {e}"
                f"You can try to install it manually, refer to `https://github.com/phiresky/ripgrep-all`."
            )
            raise

    @staticmethod
    def init_rga():
        """
        Ensure ripgrep-all is installed. If not, install it.
        """
        from agentic_search.utils.installation_utils import install_rga

        install_rga()

    def retrieve(
        self,
        terms: Union[str, List[str]],
        path: Optional[str] = None,
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
    ) -> List[Dict[str, Any]]:
        """Search for terms in files using ripgrep-all, supporting AND/OR/NOT logic.

        Args:
            terms: Single pattern (str) or list of patterns (List[str]).
            path: Path to search in (defaults to current directory).
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
            results, retrieve_pattern = self._retrieve_or(
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
            )
        elif logic == "and":
            results = self._retrieve_and(
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
            )
        elif logic == "not":
            if len(terms) < 2:
                raise ValueError(
                    "logic='not' requires at least two terms: [positive, negative1, ...]"
                )
            results = self._retrieve_not(
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
    def _run_rg(
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
            # Parse JSON Lines if requested
            if json_output and result.returncode in (0, 1) and result.stdout.strip():
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
    def _retrieve_single(**kwargs) -> List[Dict[str, Any]]:
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

        args.extend(["--rga-cache-max-blob-len", str(rga_cache_max_blob_len)])

        if rga_cache_path:
            args.extend(["--rga-cache-path", str(rga_cache_path)])

        args.append(pattern)

        if path is not None:
            if isinstance(path, (str, Path)):
                args.append(str(path))
            elif isinstance(path, list):
                for p in path:
                    args.append(str(p))
            else:
                raise TypeError(f"Unsupported type for 'path': {type(path)}")

        result = GrepRetriever._run_rg(args, json_output=not count_only)

        if result.returncode == 0:
            if count_only:
                counts = []
                for line in result.stdout.strip().splitlines():
                    if ":" in line:
                        p, c = line.rsplit(":", 1)
                        counts.append({"path": p, "count": int(c)})
                return counts
            else:
                return result.stdout
        elif result.returncode == 1:
            return []
        else:
            raise RuntimeError(
                f"ripgrep-all failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    @staticmethod
    def _retrieve_or(
        terms: List[str],
        **kwargs,
    ) -> (List[Dict[str, Any]], str):
        """OR: Match any term — simply concatenate with | (ripgrep-all supports alternation)."""
        # Escape terms if literal mode
        literal = kwargs.get("literal", False)
        if literal:
            escaped_terms = [
                term.replace("\\", "\\\\").replace("|", "\\|") for term in terms
            ]
            pattern = "|".join(escaped_terms)
            kwargs["literal"] = True  # still use -F to avoid regex meta
        else:
            # Wrap each term in (?:...) to avoid precedence issues
            pattern = "|".join(f"(?:{term})" for term in terms)
            kwargs["literal"] = False

        return GrepRetriever._retrieve_single(pattern=pattern, **kwargs), pattern

    @staticmethod
    def _retrieve_and(
        terms: List[str],
        match_same_line: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """AND: All terms in same file (default) or same line."""
        count_only = kwargs.get("count_only", False)

        # Step 1: Get files containing first term
        first_matches = GrepRetriever._retrieve_single(pattern=terms[0], **kwargs)
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
                    res = GrepRetriever._retrieve_single(
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
                matches = GrepRetriever._retrieve_single(**kwargs)
                all_matches.extend(matches)
            return all_matches

    @staticmethod
    def _retrieve_not(
        positive: str,
        negatives: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """NOT: Match positive term, but exclude files/lines containing any negative term."""
        # count_only = kwargs.get("count_only", False)

        # Step 1: Get matches for positive term
        pos_matches = GrepRetriever._retrieve_single(pattern=positive, **kwargs)
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
                res = GrepRetriever._retrieve_single(
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

    def list_files(
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

        result = GrepRetriever._run_rg(args, json_output=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"ripgrep-all --files failed: {result.stderr.strip()}")
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

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

    def replace(
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

        result = GrepRetriever._run_rg(args)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"ripgrep-all replace failed: {result.stderr.strip()}")
        return result.stdout

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
