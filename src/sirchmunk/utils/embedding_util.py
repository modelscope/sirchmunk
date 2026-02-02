# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Local Embedding Utility
Provides embedding computation using SentenceTransformer models loaded from ModelScope
"""

import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np
from loguru import logger


class EmbeddingUtil:
    """
    Embedding utility using SentenceTransformer models.
    Loads models from ModelScope for embedding computation.
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
        
        Args:
            model_id: ModelScope model identifier
            device: Device for inference ("cuda", "cpu", or None for auto-detection)
            cache_dir: Optional cache directory for model files
        """
        self.model_id = model_id
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        model_dir = self._load_model(model_id, cache_dir)
        
        # Import SentenceTransformer here to avoid dependency issues
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir, device=self.device)
        
        # Warm up model with dummy inference
        self._warmup()
        
        logger.info(
            f"Loaded embedding model: {model_id} "
            f"(device={self.device}, dim={self.dimension})"
        )
    
    @staticmethod
    def _load_model(model_id: str, cache_dir: Optional[str] = None) -> str:
        """
        Load the embedding model from ModelScope or Hugging Face.
        
        Args:
            model_id: Model identifier
            cache_dir: Optional cache directory for model files
        
        Returns:
            Path to downloaded model directory
        """
        logger.info(f"Loading embedding model: {model_id}...")
        
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
            logger.info(f"Model loaded successfully: {model_dir}")
            return model_dir
        
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(
                f"Model loading failed. Please check network or model_id. Error: {e}"
            )
    
    def _warmup(self):
        """Warm up model with dummy inference to avoid first-call latency"""
        try:
            dummy_text = ["warmup text"]
            _ = self.model.encode(dummy_text, show_progress_bar=False)
            logger.debug("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
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
        
        # SentenceTransformer.encode is CPU-bound, run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            self._encode_sync,
            texts
        )
        
        return embeddings.tolist()
    
    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous encoding wrapper for thread pool execution.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,  # Enable L2 normalization for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for MiniLM-L12-v2)"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
        }


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash for text content.
    
    Args:
        text: Input text
    
    Returns:
        Hex string of hash (first 16 characters)
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
