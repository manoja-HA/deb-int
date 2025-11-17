"""
Customer behavior embedding vector store using FAISS
"""

import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path
import logging

from ..config import config
from ..state import CustomerProfile

logger = logging.getLogger(__name__)

class CustomerEmbeddingStore:
    """FAISS-based vector store for customer behavior embeddings"""

    def __init__(self, dimension: int = None):
        """
        Initialize customer embedding store

        Args:
            dimension: Embedding dimension (default from config)
        """
        self.dimension = dimension or config.embedding_dimension
        self.index = None
        self.customer_ids: List[str] = []
        self.customer_metadata: Dict[str, Dict] = {}

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize FAISS index based on configuration"""
        if config.faiss_index_type == "flat":
            # Flat L2 index (exact search, slower for large datasets)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Initialized FAISS Flat L2 index")

        elif config.faiss_index_type == "ivf":
            # IVF index (approximate search, faster)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                config.faiss_nlist  # Number of clusters
            )
            logger.info(f"Initialized FAISS IVF index with {config.faiss_nlist} clusters")

        elif config.faiss_index_type == "hnsw":
            # HNSW index (hierarchical navigable small world, very fast)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            logger.info("Initialized FAISS HNSW index")

        else:
            raise ValueError(f"Unknown FAISS index type: {config.faiss_index_type}")

    def add_customer_embedding(
        self,
        customer_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a customer embedding to the index

        Args:
            customer_id: Customer ID
            embedding: Embedding vector
            metadata: Optional metadata about the customer
        """
        # Ensure embedding is 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # For IVF index, train if not already trained
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.warning("IVF index not trained. Training with current data...")
            self.index.train(embedding)

        # Add to index
        self.index.add(embedding.astype('float32'))
        self.customer_ids.append(customer_id)

        # Store metadata
        if metadata:
            self.customer_metadata[customer_id] = metadata

        logger.debug(f"Added embedding for customer {customer_id}")

    def add_batch_embeddings(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        metadata_dict: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Add multiple customer embeddings in batch

        Args:
            embeddings_dict: Dictionary mapping customer_id to embedding
            metadata_dict: Optional dictionary mapping customer_id to metadata
        """
        customer_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[cid] for cid in customer_ids])

        # Train IVF index if needed
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            self.index.train(embeddings.astype('float32'))

        # Add all embeddings
        self.index.add(embeddings.astype('float32'))
        self.customer_ids.extend(customer_ids)

        # Store metadata
        if metadata_dict:
            self.customer_metadata.update(metadata_dict)

        logger.info(f"Added {len(embeddings)} customer embeddings to index")

    def search_similar_customers(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar customers

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Similarity threshold (filter by distance)

        Returns:
            List of (customer_id, similarity_score) tuples
        """
        if top_k is None:
            top_k = config.similarity_top_k

        if threshold is None:
            threshold = config.similarity_threshold

        # Ensure embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Set nprobe for IVF index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = config.faiss_nprobe

        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Convert distances to similarity scores
        # For L2 distance, convert to similarity (lower distance = higher similarity)
        # Using exponential decay: similarity = exp(-distance)
        similarities = np.exp(-distances[0])

        # Filter by threshold and create results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.customer_ids) and similarity >= threshold:
                customer_id = self.customer_ids[idx]
                results.append((customer_id, float(similarity)))

        logger.info(f"Found {len(results)} similar customers above threshold {threshold}")
        return results

    def get_customer_metadata(self, customer_id: str) -> Optional[Dict]:
        """
        Get metadata for a customer

        Args:
            customer_id: Customer ID

        Returns:
            Metadata dictionary or None
        """
        return self.customer_metadata.get(customer_id)

    def save(self, filepath: str = None) -> None:
        """
        Save index and metadata to disk

        Args:
            filepath: Path to save index (default from config)
        """
        if filepath is None:
            filepath = config.vector_index_path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(filepath))

        # Save customer IDs and metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'customer_ids': self.customer_ids,
                'customer_metadata': self.customer_metadata
            }, f)

        logger.info(f"Saved customer embedding index to {filepath}")

    def load(self, filepath: str = None) -> None:
        """
        Load index and metadata from disk

        Args:
            filepath: Path to load index from (default from config)
        """
        if filepath is None:
            filepath = config.vector_index_path

        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Index file not found: {filepath}")
            return

        # Load FAISS index
        self.index = faiss.read_index(str(filepath))

        # Load customer IDs and metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.customer_ids = data['customer_ids']
                self.customer_metadata = data['customer_metadata']

        logger.info(f"Loaded customer embedding index from {filepath}")

    def get_index_size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal if self.index else 0

    def clear(self) -> None:
        """Clear the index"""
        self._initialize_index()
        self.customer_ids.clear()
        self.customer_metadata.clear()
        logger.info("Cleared customer embedding index")
