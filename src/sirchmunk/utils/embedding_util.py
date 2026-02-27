# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Local Embedding Utility
Provides embedding computation using SentenceTransformer models loaded from ModelScope
"""
import asyncio
import concurrent.futures
import hashlib
import threading
import warnings
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from loguru import logger


class EmbeddingUtil:
    """
    Embedding utility using SentenceTransformer models.
    Loads models from ModelScope for embedding computation.

    Model loading runs in a background thread so it never blocks the
    asyncio event loop.  Call ``await _ensure_model_async()`` (or the
    synchronous ``_ensure_model()``) before using ``self.model``.
    """
    
    DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize local embedding client.
        
        The heavy ``SentenceTransformer`` construction is offloaded to a
        daemon thread; ``__init__`` returns immediately.
        
        Args:
            model_id: ModelScope model identifier
            device: Device for inference ("cuda", "cpu", or None for auto-detection)
            cache_dir: Optional cache directory for model files
        """
        self.model_id = model_id
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._model_future: concurrent.futures.Future = concurrent.futures.Future()

        _worker = threading.Thread(
            target=self._load_model_bg,
            args=(model_id, cache_dir),
            daemon=True,
        )
        _worker.start()

    # ------------------------------------------------------------------
    # Background model loading
    # ------------------------------------------------------------------

    def _load_model_bg(self, model_id: str, cache_dir: Optional[str]) -> None:
        """Runs in a daemon thread: download, construct, and warm-up the model."""
        try:
            model_dir = self._download_model(model_id, cache_dir)

            from sentence_transformers import SentenceTransformer
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*position_ids.*")
                warnings.filterwarnings("ignore", category=FutureWarning)
                model = SentenceTransformer(model_dir, device=self.device)

            # Warm up with dummy inference
            model.encode(["warmup"], show_progress_bar=False)

            self.model = model
            self._model_future.set_result(model)
            logger.info(
                f"Embedding model ready: {model_id} "
                f"(device={self.device}, dim={model.get_sentence_embedding_dimension()})"
            )
        except Exception as e:
            self._model_future.set_exception(e)
            logger.error(f"Embedding model failed to load: {e}")

    @staticmethod
    def _download_model(model_id: str, cache_dir: Optional[str] = None) -> str:
        """
        Download model weights from ModelScope (or HuggingFace fallback).

        Returns:
            Path to downloaded model directory
        """
        try:
            from modelscope import snapshot_download

            model_dir = snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir,
                ignore_patterns=[
                    "openvino/*", "onnx/*", "pytorch_model.bin",
                    "rust_model.ot", "tf_model.h5"
                ]
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

    def _ensure_model(self, timeout: float = 300) -> "SentenceTransformer":
        """Block the calling thread until the model is ready."""
        if self.model is not None:
            return self.model
        return self._model_future.result(timeout=timeout)

    async def _ensure_model_async(self, timeout: float = 300) -> "SentenceTransformer":
        """Non-blocking await for model readiness (safe for the event loop)."""
        if self.model is not None:
            return self.model
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._model_future.result, timeout
        )

    def is_ready(self) -> bool:
        """Return True if the model has finished loading."""
        return self.model is not None

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

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._encode_sync, texts
        )
        return embeddings.tolist()
    
    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous encoding wrapper for thread pool execution."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for MiniLM-L12-v2)"""
        return self._ensure_model().get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata.  Blocks until the model is ready.
        
        Returns:
            Dictionary with model information
        """
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
        """
        Pre-download the embedding model without initializing.
        
        Args:
            cache_dir: Cache directory for model files
            model_id: Model identifier (uses default if None)

        Returns:
            Path to downloaded model directory
        """
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
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


if __name__ == '__main__':

    # Example usage

    async def main():
        embed_util = EmbeddingUtil()
        texts = ["Hello world", "ModelScope embedding"]
        embeddings = await embed_util.embed(texts)
        for text, emb in zip(texts, embeddings):
            print(f"Text: {text}\nEmbedding: {emb}\n")

    asyncio.run(main())