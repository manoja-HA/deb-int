"""
Embedding model wrapper for sentence transformers
"""

import numpy as np
from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer

from app.core.config import settings as config

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for sentence transformer embedding model"""

    def __init__(self, model_name: str = None):
        """
        Initialize embedding model

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or config.embedding_model
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings

        Args:
            texts: Single text or list of texts
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        if normalize_embeddings is None:
            normalize_embeddings = config.normalize_embeddings

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar
            )
            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

# Global instance
_embedding_model: EmbeddingModel = None

def get_embedding_model() -> EmbeddingModel:
    """
    Get singleton embedding model instance

    Returns:
        EmbeddingModel instance
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model
