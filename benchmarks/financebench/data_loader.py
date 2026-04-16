"""FinanceBench dataset loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class FinanceBenchLoader:
    """Load and validate FinanceBench JSONL data.

    Expects:
    - ``data_dir/financebench_open_source.jsonl`` – 150 QA rows
    - ``data_dir/financebench_document_information.jsonl`` – doc metadata (optional)
    - ``pdf_dir/`` – corpus of 41 SEC-filing PDFs named by ``doc_name``
    """

    _QUESTIONS_FILE = "financebench_open_source.jsonl"
    _DOC_INFO_FILE = "financebench_document_information.jsonl"

    def __init__(self, data_dir: str, pdf_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._pdf_dir = Path(pdf_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the 150 open-source questions from JSONL.

        Raises:
            FileNotFoundError: If the questions file is missing.
        """
        path = self._data_dir / self._QUESTIONS_FILE
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        items: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def load_doc_info(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata, keyed by ``doc_name``.

        Returns an empty dict when the file is absent (it is optional).
        """
        path = self._data_dir / self._DOC_INFO_FILE
        if not path.exists():
            return {}
        result: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    doc_name = obj.get("doc_name", "")
                    if doc_name:
                        result[doc_name] = obj
        return result

    def get_pdf_path(self, doc_name: str) -> Optional[str]:
        """Resolve *doc_name* to a PDF file path.

        Resolution order:
        1. ``<pdf_dir>/<doc_name>.pdf``
        2. ``<pdf_dir>/<doc_name>``  (file with no extension)
        3. Case-insensitive stem match across ``pdf_dir``
        """
        candidates = [
            self._pdf_dir / f"{doc_name}.pdf",
            self._pdf_dir / doc_name,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Case-insensitive fallback
        if self._pdf_dir.exists():
            lower = doc_name.lower()
            for f in self._pdf_dir.iterdir():
                if f.stem.lower() == lower:
                    return str(f)
        return None

    def get_unique_docs(self, questions: List[Dict[str, Any]]) -> Set[str]:
        """Extract the unique set of ``doc_name`` values from *questions*."""
        return {q["doc_name"] for q in questions if "doc_name" in q}

    def validate_corpus(
        self, questions: List[Dict[str, Any]]
    ) -> Tuple[int, List[str]]:
        """Check PDF availability for all referenced documents.

        Returns:
            A tuple of ``(found_count, missing_doc_names)``.
        """
        docs = self.get_unique_docs(questions)
        missing: list[str] = []
        found = 0
        for doc in sorted(docs):
            if self.get_pdf_path(doc):
                found += 1
            else:
                missing.append(doc)
        return found, missing
