# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight cross-document topic map built from compiled tree titles.

This artifact contains no embeddings and requires no external index service.
It is a compact projection of already-compiled document structure, allowing
query entities and concepts to discover related files through section titles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


# General Unicode tokenizer used when the caller does not inject one. It is a
# language-agnostic fallback (alphanumeric runs + CJK runs), NOT a curated word
# list. Discriminative-ness is decided by corpus statistics (document
# frequency) at build time rather than by any fixed stopword set, so the map
# adapts to whatever domain and languages the corpus actually contains.
_DEFAULT_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}"
)

Tokenizer = Callable[[str], Set[str]]


def _default_tokenize(text: str) -> Set[str]:
    """Tokenize a title into lowercase units plus CJK n-grams.

    No stopword filtering happens here; non-discriminative tokens are pruned
    from the built map by corpus document frequency instead.
    """
    tokens: Set[str] = set()
    for match in _DEFAULT_TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for width in (2, 3):
                tokens.update(
                    token[index:index + width]
                    for index in range(len(token) - width + 1)
                )
    return tokens


@dataclass(frozen=True)
class TopicReference:
    """One structural topic occurrence in a compiled document."""

    file_path: str
    section_title: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "file_path": self.file_path,
            "section_title": self.section_title,
        }


@dataclass
class CorpusTopicMap:
    """Token-to-document map derived from tree node titles."""

    topic_to_files: Dict[str, List[TopicReference]] = field(default_factory=dict)
    version: str = "1.0"
    tokenizer: Optional[Tokenizer] = field(
        default=None, compare=False, repr=False,
    )

    def _tokens(self, text: str) -> Set[str]:
        """Tokenize text with the injected tokenizer or the general default."""
        return (self.tokenizer or _default_tokenize)(text)

    @classmethod
    def build_from_indexer(
        cls,
        tree_indexer: Any,
        file_paths: Iterable[str],
        *,
        max_postings_per_topic: int = 500,
        tokenizer: Optional[Tokenizer] = None,
        stop_document_fraction: float = 0.5,
        min_documents_for_pruning: int = 8,
    ) -> "CorpusTopicMap":
        """Build a topic map from cached document trees.

        Non-discriminative tokens are removed by corpus document frequency
        rather than a fixed stopword list: once the corpus is large enough
        (``min_documents_for_pruning``), any token appearing in at least
        ``stop_document_fraction`` of the documents is dropped. This adapts to
        the corpus's own languages and domain vocabulary.
        """
        tokenize = tokenizer or _default_tokenize
        mapping: Dict[str, List[TopicReference]] = {}
        seen: Dict[str, Set[Tuple[str, str]]] = {}
        document_frequency: Dict[str, Set[str]] = {}
        total_documents = 0

        for file_path in file_paths:
            try:
                tree = tree_indexer.load_tree(file_path)
            except Exception:
                continue
            if tree is None or tree.root is None:
                continue
            total_documents += 1
            file_key = str(file_path)
            stack = [tree.root]
            while stack:
                node = stack.pop()
                stack.extend(getattr(node, "children", []) or [])
                title = str(getattr(node, "title", "") or "").strip()
                if not title or title == "Document":
                    continue
                reference = TopicReference(file_path=file_key, section_title=title)
                for token in tokenize(title):
                    document_frequency.setdefault(token, set()).add(file_key)
                    postings = mapping.setdefault(token, [])
                    topic_seen = seen.setdefault(token, set())
                    marker = (reference.file_path, reference.section_title)
                    if marker in topic_seen or len(postings) >= max_postings_per_topic:
                        continue
                    topic_seen.add(marker)
                    postings.append(reference)

        if total_documents >= min_documents_for_pruning and stop_document_fraction > 0:
            cutoff = stop_document_fraction * total_documents
            for token, files in document_frequency.items():
                if len(files) >= cutoff:
                    mapping.pop(token, None)

        return cls(topic_to_files=mapping, tokenizer=tokenizer)

    def search(
        self,
        terms: Iterable[str],
        *,
        allowed_paths: Optional[Set[str]] = None,
        top_k: int = 30,
    ) -> List[Tuple[str, float]]:
        """Return files ranked by structural-topic overlap."""
        query_tokens: Set[str] = set()
        normalized_terms: List[str] = []
        for term in terms:
            normalized = str(term).strip().lower()
            if not normalized:
                continue
            normalized_terms.append(normalized)
            query_tokens.update(self._tokens(normalized))
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}
        matched_sections: Dict[str, Set[str]] = {}
        for token in query_tokens:
            for reference in self.topic_to_files.get(token, []):
                if allowed_paths is not None and reference.file_path not in allowed_paths:
                    continue
                title_lower = reference.section_title.lower()
                exact_phrase = any(term in title_lower for term in normalized_terms)
                increment = 2.0 if exact_phrase else 1.0
                scores[reference.file_path] = scores.get(reference.file_path, 0.0) + increment
                matched_sections.setdefault(reference.file_path, set()).add(
                    reference.section_title
                )

        ranked = sorted(
            scores,
            key=lambda path: (
                -scores[path],
                -len(matched_sections.get(path, set())),
                path,
            ),
        )
        denominator = max(len(query_tokens), 1)
        return [
            (path, round(scores[path] / denominator, 4))
            for path in ranked[:max(1, top_k)]
        ]

    def save(self, path: Path) -> None:
        """Persist the map as deterministic JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "topic_to_files": {
                topic: [reference.to_dict() for reference in references]
                for topic, references in sorted(self.topic_to_files.items())
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["CorpusTopicMap"]:
        """Load a persisted map, returning ``None`` on invalid artifacts."""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_mapping = payload.get("topic_to_files", {})
            mapping = {
                str(topic): [
                    TopicReference(
                        file_path=str(item["file_path"]),
                        section_title=str(item["section_title"]),
                    )
                    for item in items
                    if isinstance(item, dict)
                    and item.get("file_path")
                    and item.get("section_title")
                ]
                for topic, items in raw_mapping.items()
                if isinstance(items, list)
            }
            return cls(
                topic_to_files=mapping,
                version=str(payload.get("version") or "1.0"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
