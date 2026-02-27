# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Direct document QA — bypasses the full search pipeline when the user
asks a document-level question (e.g., "请总结这篇文档", "summarize this file").

Intent detection is delegated to the LLM itself (language-agnostic, no
hardcoded patterns).  When the LLM confirms a whole-document operation,
the module collects files from the given paths, extracts their text
(full or sampled), and feeds the content directly into the LLM.

Public API:
    detect_doc_intent(query, llm) → Optional[str]   (operation or None)
    collect_doc_files(paths)      → List[DocFile]
    analyse_documents(...)        → Optional[str]
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import DETECT_DOC_INTENT, DIRECT_DOC_ANALYSIS

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DIRECT_LOAD_MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB per file
DIRECT_LOAD_MAX_CHARS     = 120_000             # ~30-60K tokens
SAMPLE_TARGET_CHARS       = 60_000              # target chars when sampling
MAX_DOC_FILES             = 5                   # cap to avoid loading entire directories

_DOC_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv",
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".rtf", ".odt", ".ods", ".odp",
    ".json", ".yaml", ".yml", ".xml",
}

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".log", ".tsv",
}


@dataclass
class DocFile:
    """Lightweight descriptor for a candidate document file."""
    path: str
    size: int
    extension: str
    is_text: bool


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def detect_doc_intent(
    query: str,
    llm: OpenAIChat,
    llm_usages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Use the LLM to decide whether *query* is a whole-document operation.

    Returns the operation name (e.g. ``"summarize"``, ``"translate"``)
    when the LLM classifies the query as document-level, or ``None``
    when it is a normal search/retrieval query.

    This is a lightweight call (short prompt, ``stream=False``) so it
    adds minimal latency as a pre-gate.
    """
    prompt = DETECT_DOC_INTENT.format(user_input=query)
    try:
        response = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        if llm_usages is not None:
            llm_usages.append(response.usage)

        parsed = _parse_json_response(response.content)
        if parsed and parsed.get("doc_level") is True:
            return parsed.get("op", "analyze")
    except Exception as exc:
        logger.debug(f"[DocQA] Intent detection failed, falling back: {exc}")

    return None


def collect_doc_files(
    paths: List[str],
    max_file_size: int = DIRECT_LOAD_MAX_FILE_SIZE,
    max_files: int = MAX_DOC_FILES,
) -> List[DocFile]:
    """Collect analyzable document files from *paths*.

    For each path element:
      - If it is a file with a supported extension → include it.
      - If it is a directory → include immediate children with supported
        extensions (non-recursive to avoid scanning large trees).

    Args:
        paths: Resolved search path strings.
        max_file_size: Skip files larger than this (bytes).
        max_files: Maximum number of files to collect.

    Returns:
        List of DocFile descriptors, sorted by size ascending.
    """
    collected: List[DocFile] = []
    seen: set = set()

    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            continue

        entries = [p] if p.is_file() else _list_dir_files(p)

        for fp in entries:
            ext = fp.suffix.lower()
            if ext not in _DOC_EXTENSIONS:
                continue

            resolved = str(fp.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                size = fp.stat().st_size
            except OSError:
                continue

            if size == 0 or size > max_file_size:
                continue

            collected.append(DocFile(
                path=resolved,
                size=size,
                extension=ext,
                is_text=ext in _TEXT_EXTENSIONS,
            ))
            if len(collected) >= max_files:
                break

        if len(collected) >= max_files:
            break

    collected.sort(key=lambda d: d.size)
    return collected


async def analyse_documents(
    query: str,
    doc_files: List[DocFile],
    llm: OpenAIChat,
    llm_usages: Optional[List[Dict[str, Any]]] = None,
    max_context_chars: int = DIRECT_LOAD_MAX_CHARS,
    sample_target_chars: int = SAMPLE_TARGET_CHARS,
) -> Optional[str]:
    """Extract text from *doc_files*, build prompt, and call LLM.

    If the combined text fits within *max_context_chars* the full content
    is sent; otherwise each document is sampled (head + mid + tail) to
    stay within *sample_target_chars*.

    Args:
        query: User's original question/instruction.
        doc_files: Files to analyse (from :func:`collect_doc_files`).
        llm: LLM client for chat completion.
        llm_usages: Optional list; LLM usage dicts are appended here.
        max_context_chars: Threshold above which sampling kicks in.
        sample_target_chars: Per-file char budget when sampling.

    Returns:
        LLM answer string, or None if no text could be extracted.
    """
    # ---- 1. Extract text from every file ----
    doc_contents: List[Tuple[DocFile, str]] = []
    total_chars = 0
    for df in doc_files:
        text = await _extract_text(df)
        if text:
            doc_contents.append((df, text))
            total_chars += len(text)

    if not doc_contents:
        return None

    # ---- 2. Build document section: full or sampled ----
    needs_sampling = total_chars > max_context_chars
    per_file_budget = sample_target_chars // len(doc_contents) if needs_sampling else 0

    parts: List[str] = []
    for df, text in doc_contents:
        fname = Path(df.path).name
        content = _sample_text(text, per_file_budget) if needs_sampling else text
        parts.append(f"#### File: {fname}\n```\n{content}\n```")

    documents_text = "\n\n".join(parts)

    # ---- 3. Prompt → LLM ----
    prompt = DIRECT_DOC_ANALYSIS.format(
        documents=documents_text,
        user_input=query,
    )

    response = await llm.achat(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    if llm_usages is not None:
        llm_usages.append(response.usage)

    return response.content if response.content else None


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response string."""
    text = text.strip()
    # Direct parse (ideal case)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # Last resort: find first { ... } substring
    match = re.search(r"\{[^{}]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _list_dir_files(directory: Path) -> List[Path]:
    """List immediate child files of *directory* (non-recursive)."""
    try:
        return sorted(
            (e for e in directory.iterdir() if e.is_file()),
            key=lambda f: f.name,
        )
    except PermissionError:
        return []


async def _extract_text(df: DocFile) -> Optional[str]:
    """Extract plain text from a DocFile.

    Uses direct file read for text formats; falls back to kreuzberg
    ``fast_extract`` for binary formats (PDF, DOCX, PPTX, …).
    """
    try:
        if df.is_text:
            return Path(df.path).read_text(encoding="utf-8", errors="replace")

        from sirchmunk.utils.file_utils import fast_extract
        result = await fast_extract(df.path)
        return result.content if result and result.content else None
    except Exception as exc:
        logger.debug(f"[DocQA] Text extraction failed for {df.path}: {exc}")
        return None


def _sample_text(text: str, target_chars: int) -> str:
    """Return a representative sample of *text* within *target_chars*.

    Strategy: head (40%) + middle excerpt (40%) + tail (20%).
    Markers ``[...content sampled...]`` indicate omitted regions.
    """
    if len(text) <= target_chars:
        return text

    head_size = int(target_chars * 0.40)
    tail_size = int(target_chars * 0.20)
    mid_budget = target_chars - head_size - tail_size

    head = text[:head_size]
    tail = text[-tail_size:] if tail_size > 0 else ""

    # Middle: centred excerpt
    mid_start = len(text) // 2 - mid_budget // 2
    mid_end = mid_start + mid_budget
    mid = text[max(mid_start, head_size):min(mid_end, len(text) - tail_size)]

    parts = [head]
    if mid:
        parts.append("[...content sampled...]")
        parts.append(mid)
    parts.append("[...content sampled...]")
    if tail:
        parts.append(tail)

    return "\n".join(parts)
