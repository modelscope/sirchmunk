# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for AgenticSearch.search() entry point.

Every test calls the real search() with a real LLM and real files —
no mocks, no patches.  Configuration is loaded exclusively from
tests/.env.test.

Return-value contract (after the return_cluster removal):
    - Default (return_context=False):
        - FAST / DEEP text search → ``str``
        - FILENAME_ONLY           → ``List[Dict]``
        - Vision (auto / forced)  → ``str``
    - return_context=True:
        - All modes → ``SearchContext``
            .answer  : str
            .cluster : KnowledgeCluster | None
"""

import asyncio
import json
import os
import unittest
from pathlib import Path
from typing import Dict, List

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.schema.knowledge import KnowledgeCluster
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.search import AgenticSearch

try:
    from sirchmunk.vision.vision_search import VisionSearchResult
    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False


# ------------------------------------------------------------------ #
# Test configuration — loaded from .env.test
# ------------------------------------------------------------------ #

_TESTS_DIR = Path(__file__).resolve().parent
_ENV_FILE = _TESTS_DIR / ".env.test"


def _load_env(path: Path) -> Dict[str, str]:
    """Parse a dotenv-style file into a dict (no shell expansion)."""
    cfg: Dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Test env file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip()
    return cfg


_CFG = _load_env(_ENV_FILE)


def _cfg(key: str, default: str = "") -> str:
    return _CFG.get(key, default)


def _cfg_int(key: str, default: int = 0) -> int:
    return int(_CFG.get(key, str(default)))


def _cfg_float(key: str, default: float = 0.0) -> float:
    return float(_CFG.get(key, str(default)))


def _cfg_bool(key: str, default: bool = False) -> bool:
    return _CFG.get(key, str(default)).lower() in ("true", "1", "yes")


def _cfg_list(key: str) -> List[str]:
    raw = _cfg(key)
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


# ------------------------------------------------------------------ #
# Base test class — real AgenticSearch, real LLM, real files
# ------------------------------------------------------------------ #

class _BaseSearchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api_key = _cfg("LLM_API_KEY")
        if not api_key:
            raise unittest.SkipTest("LLM_API_KEY not configured in .env.test")

        search_paths = _cfg_list("SEARCH_PATHS")
        if not search_paths:
            raise unittest.SkipTest("SEARCH_PATHS not configured in .env.test")

        cls.search_paths = search_paths
        cls.query_images = _cfg_list("QUERY_IMAGES")

        llm = OpenAIChat(
            base_url=_cfg("LLM_BASE_URL"),
            api_key=api_key,
            model=_cfg("LLM_MODEL_NAME"),
            timeout=_cfg_float("LLM_TIMEOUT", 60.0),
        )

        work_path = _cfg("SIRCHMUNK_WORK_PATH") or os.path.join(
            os.path.expanduser("~"), ".sirchmunk", "test_work",
        )

        cls.searcher = AgenticSearch(
            llm=llm,
            work_path=work_path,
            paths=search_paths,
            verbose=_cfg_bool("SIRCHMUNK_VERBOSE"),
            reuse_knowledge=_cfg_bool("SIRCHMUNK_ENABLE_CLUSTER_REUSE"),
        )

    def _run(self, coro):
        return asyncio.run(coro)


# ================================================================== #
#  FAST MODE                                                           #
# ================================================================== #

class TestSearchFastMode(_BaseSearchTest):

    def test_fast_returns_answer_string(self):
        """FAST mode returns a non-empty string answer."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
        ))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_fast_return_context(self):
        """FAST + return_context returns a SearchContext with answer and cluster."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)
        self.assertIsNotNone(result.cluster)
        self.assertIsInstance(result.cluster, KnowledgeCluster)
        self.assertTrue(result.cluster.id.startswith("FS"))
        self.assertEqual(result.answer, result.cluster.content)

    def test_fast_context_serializable(self):
        """SearchContext.to_dict() produces a JSON-serializable dict."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("answer", d)
        serialized = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(serialized, str)


# ================================================================== #
#  DEEP MODE                                                           #
# ================================================================== #

class TestSearchDeepMode(_BaseSearchTest):

    def test_deep_returns_answer_string(self):
        """DEEP mode returns a non-empty string answer."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
        ))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_deep_return_context(self):
        """DEEP + return_context returns a SearchContext with answer and cluster."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)
        self.assertIsNotNone(result.cluster)
        self.assertIsInstance(result.cluster, KnowledgeCluster)
        self.assertEqual(result.answer, result.cluster.content)


# ================================================================== #
#  FILENAME_ONLY MODE                                                  #
# ================================================================== #

class TestSearchFilenameOnly(_BaseSearchTest):

    def test_filename_only_returns_list(self):
        """FILENAME_ONLY returns a list of file match dicts."""
        query = _cfg("TEST_QUERY_FILENAME", "notes")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FILENAME_ONLY",
        ))

        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_filename_only_no_matches(self):
        """No matches returns an error string."""
        result = self._run(self.searcher.search(
            query="__nonexistent_file_xyz_42__",
            paths=self.search_paths,
            mode="FILENAME_ONLY",
        ))

        self.assertIsInstance(result, str)
        self.assertIn("No files found", result)


# ================================================================== #
#  PATH VALIDATION                                                     #
# ================================================================== #

class TestPathValidation(unittest.TestCase):
    """Unit tests for AgenticSearch.validate_search_paths (no LLM)."""

    def test_rejects_hyphen_prefix(self):
        clean = AgenticSearch.validate_search_paths(["--help", "/tmp"])
        self.assertEqual(len(clean), 1)
        self.assertNotIn("--help", clean)

    def test_rejects_null_byte(self):
        clean = AgenticSearch.validate_search_paths(["/tmp/foo\x00bar"])
        self.assertEqual(clean, [])

    def test_rejects_nonexistent_with_require_exists(self):
        clean = AgenticSearch.validate_search_paths(
            ["/absolutely/does/not/exist/xyz"],
            require_exists=True,
        )
        self.assertEqual(clean, [])

    def test_accepts_valid_url(self):
        clean = AgenticSearch.validate_search_paths(
            ["https://example.com/docs"],
        )
        self.assertEqual(clean, ["https://example.com/docs"])

    def test_rejects_malformed_url(self):
        clean = AgenticSearch.validate_search_paths(["https://"])
        self.assertEqual(clean, [])

    def test_deduplicates(self):
        clean = AgenticSearch.validate_search_paths(["/tmp", "/tmp", "/tmp"])
        self.assertEqual(len(clean), 1)


# ================================================================== #
#  VISION AUTO-DETECT (text query triggers vision pipeline)            #
# ================================================================== #

@unittest.skipUnless(_HAS_VISION, "Vision dependencies not installed")
class TestSearchVisionAutoDetect(_BaseSearchTest):
    """Vision search triggered by LLM-detected image intent.

    After the SearchContext unification, vision auto-detect returns a
    ``str`` answer (or ``SearchContext`` with ``return_context=True``),
    NOT a raw ``List[VisionSearchResult]``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.vision_paths = _cfg_list("TEST_VISION_SEARCH_PATHS")
        if not cls.vision_paths:
            raise unittest.SkipTest(
                "TEST_VISION_SEARCH_PATHS not configured in .env.test"
            )

    def test_vision_fast_returns_string(self):
        """FAST vision auto-detect returns a structured answer string."""
        query = _cfg(
            "TEST_QUERY_VISION_FAST",
            "find photos of dogs playing outdoors",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="FAST",
        ))
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_vision_fast_return_context(self):
        """FAST vision + return_context returns SearchContext with vision cluster."""
        query = _cfg(
            "TEST_QUERY_VISION_FAST",
            "find photos of dogs playing outdoors",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)

        if result.cluster is not None:
            self.assertEqual(result.answer, result.cluster.content)
            self.assertTrue(result.cluster.id.startswith("VS") or
                            result.cluster.id.startswith("FS"))
            for sr in result.cluster.search_results:
                if isinstance(sr, dict):
                    self.assertIn("path", sr)
                    self.assertIn("confidence", sr)

    def test_vision_fast_context_has_vlm_captions(self):
        """FAST vision now includes VLM collage verification with captions."""
        query = _cfg(
            "TEST_QUERY_VISION_FAST",
            "find photos of dogs playing outdoors",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        if result.cluster:
            self.assertEqual(result.answer, result.cluster.content)
            if result.cluster.evidences:
                has_caption = any(ev.summary for ev in result.cluster.evidences)
                self.assertTrue(
                    has_caption,
                    "FAST vision should have VLM-generated captions",
                )
            if result.cluster.search_results:
                for sr in result.cluster.search_results:
                    if isinstance(sr, dict):
                        self.assertIn("caption", sr)

    def test_vision_deep_returns_string(self):
        """DEEP vision auto-detect returns a structured answer string."""
        query = _cfg(
            "TEST_QUERY_VISION_DEEP",
            "find all images containing a red fire truck on the street",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="DEEP",
        ))
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_vision_deep_context_has_captions(self):
        """DEEP vision + return_context: evidence summaries carry VLM captions."""
        result = self._run(self.searcher.search(
            query="find images of people riding bicycles",
            paths=self.vision_paths,
            mode="DEEP",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        if result.cluster:
            self.assertEqual(result.answer, result.cluster.content)
            if result.cluster.evidences:
                has_caption = any(ev.summary for ev in result.cluster.evidences)
                self.assertTrue(
                    has_caption,
                    "At least one evidence should have a caption as summary",
                )
            if result.cluster.search_results:
                for sr in result.cluster.search_results:
                    if isinstance(sr, dict):
                        self.assertIn("path", sr)
                        self.assertIn("caption", sr)


# ================================================================== #
#  VISION DISPATCH (explicit query_images)                             #
# ================================================================== #

@unittest.skipUnless(_HAS_VISION, "Vision dependencies not installed")
class TestSearchVisionQueryImages(_BaseSearchTest):
    """Vision search with explicit reference images (image-to-image).

    With query_images, search() now returns ``str`` (or ``SearchContext``
    with ``return_context=True``), NOT raw ``List[VisionSearchResult]``.
    """

    def setUp(self):
        if not self.query_images:
            self.skipTest("QUERY_IMAGES not configured in .env.test")

    def test_query_images_returns_string(self):
        """Providing query_images returns a structured answer string."""
        result = self._run(self.searcher.search(
            query="find similar images",
            paths=self.search_paths,
            query_images=self.query_images,
            mode="FAST",
        ))

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_query_images_return_context(self):
        """query_images + return_context returns SearchContext with vision cluster."""
        result = self._run(self.searcher.search(
            query="find similar images",
            paths=self.search_paths,
            query_images=self.query_images,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        if result.cluster:
            self.assertEqual(result.answer, result.cluster.content)
            self.assertTrue(result.cluster.id.startswith("VS"))
            for sr in result.cluster.search_results:
                self.assertIsInstance(sr, dict)
                self.assertIn("path", sr)

    def test_query_images_deep_return_context(self):
        """DEEP mode + query_images + return_context."""
        result = self._run(self.searcher.search(
            query="find similar images",
            paths=self.search_paths,
            query_images=self.query_images,
            mode="DEEP",
            return_context=True,
        ))

        self.assertIsInstance(result, SearchContext)
        self.assertIsInstance(result.answer, str)
        if result.cluster:
            self.assertEqual(result.answer, result.cluster.content)


# ================================================================== #
#  VisionSearchResult unit tests                                       #
# ================================================================== #

@unittest.skipUnless(_HAS_VISION, "Vision dependencies not installed")
class TestVisionSearchResult(unittest.TestCase):
    """Unit tests for VisionSearchResult serialization (no LLM)."""

    def _make(self, **overrides):
        defaults = dict(
            path="/tmp/test.jpg",
            caption="A dog in a park",
            confidence=0.85,
            semantic_tags=["dog", "park", "outdoor"],
            source="fast_pipeline",
        )
        defaults.update(overrides)
        return VisionSearchResult(**defaults)

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        for key in ("path", "caption", "confidence", "semantic_tags", "source"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        r = self._make()
        d = r.to_dict()
        self.assertEqual(d["path"], "/tmp/test.jpg")
        self.assertEqual(d["caption"], "A dog in a park")
        self.assertAlmostEqual(d["confidence"], 0.85, places=2)
        self.assertEqual(d["semantic_tags"], ["dog", "park", "outdoor"])
        self.assertEqual(d["source"], "fast_pipeline")

    def test_to_dict_json_serializable(self):
        r = self._make()
        serialized = json.dumps(r.to_dict())
        self.assertIsInstance(serialized, str)

    def test_str_contains_path(self):
        r = self._make()
        s = str(r)
        self.assertIn("/tmp/test.jpg", s)
        self.assertIn("VisionSearchResult", s)

    def test_str_without_caption(self):
        r = self._make(caption="")
        s = str(r)
        self.assertIn("/tmp/test.jpg", s)
        self.assertNotIn("caption=", s)


if __name__ == "__main__":
    unittest.main()
