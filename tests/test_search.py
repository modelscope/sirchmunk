# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for AgenticSearch.search() entry point.

Every test calls the real search() with a real LLM and real files —
no mocks, no patches.  Configuration is loaded exclusively from
tests/.env.test.
"""

import asyncio
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
        return asyncio.get_event_loop().run_until_complete(coro)


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

    def test_fast_return_cluster(self):
        """FAST + return_cluster returns a KnowledgeCluster."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_cluster=True,
        ))

        self.assertIsInstance(result, KnowledgeCluster)
        self.assertTrue(result.id.startswith("FS"))
        self.assertTrue(len(result.content) > 0)

    def test_fast_return_context(self):
        """FAST + return_context returns (answer, SearchContext) tuple."""
        query = _cfg("TEST_QUERY_FAST", "transformer attention mechanism")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="FAST",
            return_context=True,
        ))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], SearchContext)


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

    def test_deep_return_cluster(self):
        """DEEP + return_cluster returns a KnowledgeCluster."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
            return_cluster=True,
        ))

        self.assertIsInstance(result, KnowledgeCluster)
        self.assertTrue(len(result.id) > 0)

    def test_deep_return_context(self):
        """DEEP + return_context returns (answer, SearchContext) tuple."""
        query = _cfg("TEST_QUERY_DEEP", "How does transformer attention work?")
        result = self._run(self.searcher.search(
            query=query,
            paths=self.search_paths,
            mode="DEEP",
            return_context=True,
        ))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], SearchContext)


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
#  VISION AUTO-DETECT (text query triggers vision pipeline)            #
# ================================================================== #

@unittest.skipUnless(_HAS_VISION, "Vision dependencies not installed")
class TestSearchVisionAutoDetect(_BaseSearchTest):
    """Vision search triggered by LLM-detected image intent (COCO2017)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.vision_paths = _cfg_list("TEST_VISION_SEARCH_PATHS")
        if not cls.vision_paths:
            raise unittest.SkipTest(
                "TEST_VISION_SEARCH_PATHS not configured in .env.test"
            )

    def _assert_vision_results(self, results: list):
        """Shared assertion helper for VisionSearchResult lists."""
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0, "Expected at least one vision result")
        for r in results:
            self.assertIsInstance(r, VisionSearchResult)
            self.assertTrue(
                os.path.isfile(r.path),
                f"Result path does not exist: {r.path}",
            )
            self.assertGreaterEqual(r.confidence, 0.0)
            self.assertTrue(len(r.path) > 0)

    def test_vision_fast_returns_results(self):
        """FAST vision: COCO2017 query returns VisionSearchResult list."""
        query = _cfg(
            "TEST_QUERY_VISION_FAST",
            "find photos of dogs playing outdoors",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="FAST",
        ))
        self._assert_vision_results(result)
        for r in result:
            self.assertEqual(r.source, "fast_pipeline")

    def test_vision_deep_returns_results(self):
        """DEEP vision: COCO2017 query returns verified VisionSearchResult."""
        query = _cfg(
            "TEST_QUERY_VISION_DEEP",
            "find all images containing a red fire truck on the street",
        )
        result = self._run(self.searcher.search(
            query=query,
            paths=self.vision_paths,
            mode="DEEP",
        ))
        self._assert_vision_results(result)

    def test_vision_result_has_caption(self):
        """DEEP vision results should carry VLM-generated captions."""
        result = self._run(self.searcher.search(
            query="find images of people riding bicycles",
            paths=self.vision_paths,
            mode="DEEP",
        ))
        self._assert_vision_results(result)
        captioned = [r for r in result if r.caption]
        self.assertTrue(
            len(captioned) > 0,
            "At least one DEEP result should have a VLM caption",
        )


# ================================================================== #
#  VISION DISPATCH (explicit query_images)                             #
# ================================================================== #

@unittest.skipUnless(_HAS_VISION, "Vision dependencies not installed")
class TestSearchVisionQueryImages(_BaseSearchTest):
    """Vision search with explicit reference images (image-to-image)."""

    def setUp(self):
        if not self.query_images:
            self.skipTest("QUERY_IMAGES not configured in .env.test")

    def test_query_images_forces_vision(self):
        """Providing query_images triggers the vision pipeline."""
        result = self._run(self.searcher.search(
            query="find similar images",
            paths=self.search_paths,
            query_images=self.query_images,
            mode="FAST",
        ))

        self.assertIsInstance(result, list)
        if result:
            self.assertIsInstance(result[0], VisionSearchResult)

    def test_query_images_deep_mode(self):
        """DEEP mode with query_images returns verified results."""
        result = self._run(self.searcher.search(
            query="find similar images",
            paths=self.search_paths,
            query_images=self.query_images,
            mode="DEEP",
        ))

        self.assertIsInstance(result, list)
        if result:
            for r in result:
                self.assertIsInstance(r, VisionSearchResult)
                self.assertTrue(os.path.isfile(r.path))


if __name__ == "__main__":
    unittest.main()
