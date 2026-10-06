# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Local Embedding Utility
Provides embedding computation using SentenceTransformer models loaded from ModelScope.

Model loading is **lazy**: the heavy ``SentenceTransformer`` construction
only starts when ``start_loading()`` is called, and runs in a daemon
thread so it never blocks the asyncio event loop or contends for the GIL
during time-critical search operations.
"""
import asyncio
import concurrent.futures
import hashlib
import os
import threading
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from loguru import logger
from sirchmunk.storage.lifecycle import get_runtime


def _managed_cache(runtime, cache_dir=None):
    """All writable library caches stay in the explicitly owned work root."""
    cache = Path(cache_dir or runtime.root / ".cache" / "models").expanduser().resolve()
    if not cache.is_relative_to(runtime.root):
        runtime.fail("embedding_cache_outside_managed_root")
        raise ValueError("Managed embedding cache must be inside the work root")
    return cache


def _configure_managed_libraries(cache, runtime):
    # Set before importing torch/transformers/model hubs. Explicit loader cache
    # parameters are also supplied in case another module imported them earlier.
    variables = {
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface" / "hub"),
        "SENTENCE_TRANSFORMERS_HOME": str(cache),
        "MODELSCOPE_CACHE": str(cache / "modelscope"),
        "TORCH_HOME": str(cache / "torch"),
        "HF_MODULES_CACHE": str(cache / "huggingface" / "modules"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for key, value in variables.items():
        if value != "1" and not Path(value).resolve().is_relative_to(cache):
            runtime.fail("linked_embedding_cache_outside_managed_root")
            raise ValueError("Managed library cache resolves outside the configured cache")
    os.environ.update(variables)


def get_embedding_dim() -> int:
    """Configured embedding dimension (env ``EMBEDDING_DIM``, default 384).

    Must match the active embedding model output dimension, e.g. 1024 for
    ``microsoft/harrier-oss-v1-0.6b``.  Consumers that persist vectors with a
    fixed-size type (DuckDB ``FLOAT[n]``) read the dimension from here.
    """
    value = os.getenv("EMBEDDING_DIM", "384")
    try:
        dimension = int(value)
    except ValueError as exc:
        raise ValueError("EMBEDDING_DIM must be a positive integer") from exc
    if not 1 <= dimension <= 65536:
        raise ValueError("EMBEDDING_DIM must be between 1 and 65536")
    return dimension


class EmbeddingUtil:
    """Embedding utility using SentenceTransformer models (lazy-loaded).

    Construction is cheap: ``__init__`` only stores configuration.
    Call ``start_loading()`` to kick off the background model download +
    construction, and ``is_ready()`` to check completion.  ``embed()``
    will call ``start_loading()`` automatically if it hasn't been called.
    """

    DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id or os.getenv("EMBEDDING_MODEL_ID", self.DEFAULT_MODEL_ID)
        self._runtime = get_runtime()
        self.expected_dimension = get_embedding_dim()
        self._cache_dir = cache_dir or os.getenv("EMBEDDING_CACHE_DIR")
        if self._cache_dir:
            self._cache_dir = os.path.expanduser(self._cache_dir)
        if self._runtime:
            self._cache_dir = str(_managed_cache(self._runtime, self._cache_dir))
            _configure_managed_libraries(Path(self._cache_dir), self._runtime)
        self.model = None

        # Defer torch import to the background thread to avoid blocking
        # the event loop with the heavy first-import cost.
        if device is not None:
            self.device = device
        elif self._runtime:
            self.device = None  # Resolve under the model-loading thread's lease.
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        self._model_future: concurrent.futures.Future = concurrent.futures.Future()
        self._loading_started = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy background loading
    # ------------------------------------------------------------------

    def start_loading(self) -> None:
        """Kick off the background model load (idempotent)."""
        with self._lock:
            if self._loading_started:
                return
            runtime = self._runtime
            # Admission rejection is not a failed model load; it may be retried
            # after maintenance resumes. No future/thread is created on refusal.
            release = runtime.reserve_background() if runtime else None
            try:
                self._loading_started = True

                def load():
                    try:
                        self._load_model_bg(self.model_id, self._cache_dir)
                    finally:
                        if release:
                            release()

                worker = threading.Thread(target=load, daemon=True,
                                          name="sirchmunk-embedding-load")
                worker.start()
            except BaseException as exc:
                if runtime:
                    runtime.fail("embedding_load_dispatch_failed")
                if release:
                    release()
                if not self._model_future.done():
                    self._model_future.set_exception(exc)
                raise
        logger.info("Embedding model loading started in background thread")

    def _load_model_bg(self, model_id: str, cache_dir: Optional[str]) -> None:
        """Runs in a daemon thread: download, construct, and warm-up."""
        try:
            if self._runtime:
                model_dir = self._managed_model_path(model_id, cache_dir, self._runtime)
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                if self.device is None:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                model_dir = self._download_model(model_id, cache_dir)

            from sentence_transformers import SentenceTransformer
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*position_ids.*")
                warnings.filterwarnings("ignore", category=FutureWarning)
                if self._runtime:
                    model = SentenceTransformer(
                        model_dir, device=self.device, cache_folder=cache_dir,
                        local_files_only=True, trust_remote_code=False,
                    )
                else:
                    model = SentenceTransformer(model_dir, device=self.device)

            if model.get_sentence_embedding_dimension() != self.expected_dimension:
                raise ValueError("Embedding model output does not match EMBEDDING_DIM")
            model.encode(["warmup"], show_progress_bar=False)

            self.model = model
            self._model_future.set_result(model)
            logger.info(
                f"Embedding model ready: {model_id} "
                f"(device={self.device}, dim={model.get_sentence_embedding_dimension()})"
            )
        except BaseException as e:
            if self._runtime:
                self._runtime.fail("embedding_model_load_failed")
            if not self._model_future.done():
                self._model_future.set_exception(e)
            logger.error(f"Embedding model failed to load: {e}")

    @staticmethod
    def _managed_model_path(model_id, cache_dir, runtime):
        """Resolve preprovisioned local/ModelScope/HF files without any hub call."""
        cache = _managed_cache(runtime, cache_dir)
        direct = Path(model_id).expanduser()
        if direct.is_dir():
            candidates = [direct.resolve()]
        else:
            # Only safe repository identifiers may be mapped beneath a cache.
            parts = model_id.replace("\\", "/").split("/")
            if any(part in {"", ".", ".."} or ":" in part for part in parts):
                raise RuntimeError("Managed embedding model path is not provisioned")
            candidates = []
            for base in (cache, cache / "models", cache / "modelscope" / "models",
                         cache / "hub" / "models"):
                candidate = base.joinpath(*parts).resolve()
                if candidate.is_relative_to(cache):
                    candidates.append(candidate)
            for base in (cache, cache / "hub", cache / "huggingface" / "hub"):
                repo = base / ("models--" + "--".join(parts))
                reference = repo / "refs" / "main"
                if reference.is_file() and reference.stat().st_size <= 256:
                    revision = reference.read_text(encoding="utf-8").strip()
                    if revision and all(char in "0123456789abcdefABCDEF" for char in revision):
                        candidate = (repo / "snapshots" / revision).resolve()
                        if candidate.is_relative_to(cache):
                            candidates.append(candidate)
        for candidate in candidates:
            if candidate.is_dir() and any((candidate / name).is_file()
                                         for name in ("modules.json", "config.json")):
                return str(candidate)
        raise RuntimeError(
            "Managed embedding model is not provisioned locally; "
            "provide a model directory or an existing cache before startup. "
            "Online downloads are disabled."
        )

    @staticmethod
    def _download_model(model_id: str, cache_dir: Optional[str] = None) -> str:
        """Download model weights from ModelScope (or HuggingFace fallback)."""
        runtime = get_runtime()
        if runtime:
            try:
                return EmbeddingUtil._managed_model_path(model_id, cache_dir, runtime)
            except Exception:
                runtime.fail("embedding_model_not_provisioned")
                raise
        try:
            from modelscope import snapshot_download
        except ImportError:
            raise RuntimeError("modelscope is required. Install with: pip install modelscope")

        ignore = ["openvino/*", "onnx/*", "pytorch_model.bin", "rust_model.ot", "tf_model.h5"]

        # Step 1: offline-first — try loading from local cache
        try:
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir,
                local_files_only=True,
                ignore_patterns=ignore,
            )
            logger.info(f"Model loaded from cache: {model_dir}")
            return model_dir
        except Exception:
            logger.debug(f"Offline cache miss for {model_id}, trying online download...")

        # Step 2: fallback — online download
        try:
            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
                ignore_patterns=ignore,
            )
            logger.debug(f"Model downloaded: {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise RuntimeError(
                f"Model download failed. Please check network or model_id. Error: {e}"
            )

    # ------------------------------------------------------------------
    # Model readiness helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the model has finished loading."""
        return self.model is not None

    def _ensure_model(self, timeout: float = 300) -> "SentenceTransformer":
        """Block the calling thread until the model is ready."""
        if self.model is not None:
            return self.model
        self.start_loading()
        return self._model_future.result(timeout=timeout)

    async def _ensure_model_async(self, timeout: float = 300) -> "SentenceTransformer":
        """Non-blocking await for model readiness (safe for the event loop)."""
        if self.model is not None:
            return self.model
        self.start_loading()
        if self._runtime:
            future = asyncio.wrap_future(self._model_future)
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._model_future.result, timeout
        )

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for batch texts using local model.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors (each of dimension 384)
        """
        if not texts:
            return []

        await self._ensure_model_async()

        if self._runtime:
            from sirchmunk.llm.factory import run_background_thread
            embeddings = await run_background_thread(self._encode_sync, texts, runtime=self._runtime)
        else:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._encode_sync, texts)
        return embeddings.tolist()

    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous encoding wrapper for thread pool execution."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    @property
    def query_instruction(self) -> Optional[str]:
        """Instruction prefix for query-side embeddings (asymmetric retrieval).

        Only instruction-aware models (e.g. microsoft/harrier-oss-v1) need this.
        Configured via ``EMBEDDING_QUERY_INSTRUCTION``; when unset, returns None
        and ``embed_query`` degrades to plain ``embed`` (bge-m3 / MiniLM style).
        The ``{query}`` placeholder, if present, is replaced with the query text.
        dotenv does not interpret escape sequences, so a literal ``\\n`` written in
        ``.env`` is restored to a real newline here (the harrier instruction format
        requires ``Instruct: ...\\nQuery: `` with a genuine line break).
        """
        instr = os.getenv("EMBEDDING_QUERY_INSTRUCTION") or None
        if instr:
            instr = instr.replace("\\n", "\n")
        return instr

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single *query* with the retrieval instruction prefix.

        Document-side text must keep using :meth:`embed` (no prefix) so that
        the asymmetric query/document encoding the model was trained on holds.
        """
        instr = self.query_instruction
        if not instr:
            return (await self.embed([query]))[0]
        text = instr.replace("{query}", query) if "{query}" in instr else f"{instr}{query}"
        return (await self.embed([text]))[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for MiniLM-L12-v2)."""
        return self._ensure_model().get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.  Blocks until the model is ready."""
        model = self._ensure_model()
        return {
            "model_id": self.model_id,
            "dimension": model.get_sentence_embedding_dimension(),
            "device": self.device,
            "max_seq_length": model.max_seq_length,
        }

    @classmethod
    def preload_model(
        cls,
        cache_dir: Optional[str] = None,
        model_id: str = None,
    ) -> str:
        """Pre-download the embedding model without initializing."""
        model_id = model_id or cls.DEFAULT_MODEL_ID
        return cls._download_model(model_id, cache_dir)


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash for text content.

    Args:
        text: Input text

    Returns:
        Hex string of hash (first 16 characters)
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":

    async def main():
        embed_util = EmbeddingUtil()
        texts = ["Hello world", "ModelScope embedding"]
        embeddings = await embed_util.embed(texts)
        for text, emb in zip(texts, embeddings):
            print(f"Text: {text}\nEmbedding: {emb}\n")

    asyncio.run(main())
