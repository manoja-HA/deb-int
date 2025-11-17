"""Vector repository - ChromaDB vector store access"""

from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorRepository:
    """Vector store repository (singleton)"""

    _instance = None

    def __init__(self):
        self.client = None
        self.collection = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "VectorRepository":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_index(self) -> None:
        """Load ChromaDB collection"""
        if self._loaded:
            return

        try:
            chroma_dir = settings.EMBEDDINGS_DIR / "chroma"

            if not chroma_dir.exists():
                logger.warning(f"ChromaDB directory not found: {chroma_dir}")
                logger.warning("Run scripts/build_vector_index.py to create index")
                return

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )

            # Get or create collection
            from chromadb.utils import embedding_functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )

            try:
                self.collection = self.client.get_collection(
                    name="customers",
                    embedding_function=embedding_function
                )
                self._loaded = True
                logger.info(f"Loaded ChromaDB collection with {self.collection.count()} customers")
            except Exception as e:
                logger.error(f"Collection 'customers' not found: {e}")
                logger.warning("Run scripts/build_vector_index.py to create collection")

        except Exception as e:
            logger.error(f"Failed to load ChromaDB: {e}")

    def search_similar(
        self,
        query_text: str,
        top_k: int = 20,
        threshold: float = 0.75,
    ) -> List[Tuple[str, float]]:
        """Search for similar customers using behavior text

        Args:
            query_text: Customer behavior text description
            top_k: Number of similar customers to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (customer_id, similarity_score) tuples
        """
        if not self._loaded or self.collection is None:
            self.load_index()

        if self.collection is None:
            logger.warning("ChromaDB collection not available")
            return []

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["metadatas", "distances"]
            )

            # Build results list
            similar_customers = []

            if results and results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                distances = results['distances'][0]

                for customer_id, distance in zip(ids, distances):
                    # Convert distance to similarity (lower distance = higher similarity)
                    # Using exponential decay: similarity = exp(-distance)
                    # For cosine distance (0-2 range), this works well
                    similarity = max(0.0, 1.0 - (distance / 2.0))

                    if similarity >= threshold:
                        similar_customers.append((customer_id, float(similarity)))

            return similar_customers

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_metadata(self, customer_id: str) -> Optional[Dict]:
        """Get customer metadata from ChromaDB

        Args:
            customer_id: Customer ID

        Returns:
            Customer metadata dictionary or None
        """
        if not self._loaded or self.collection is None:
            self.load_index()

        if self.collection is None:
            return None

        try:
            results = self.collection.get(
                ids=[customer_id],
                include=["metadatas"]
            )

            if results and results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]

            return None

        except Exception as e:
            logger.error(f"Failed to get metadata for customer {customer_id}: {e}")
            return None

    def is_loaded(self) -> bool:
        """Check if collection is loaded"""
        return self._loaded and self.collection is not None
