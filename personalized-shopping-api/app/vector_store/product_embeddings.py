"""
Product embedding vector store
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from .customer_embeddings import CustomerEmbeddingStore
from ..config import config

logger = logging.getLogger(__name__)

class ProductEmbeddingStore(CustomerEmbeddingStore):
    """
    Vector store for product embeddings
    Inherits from CustomerEmbeddingStore with product-specific methods
    """

    def __init__(self, dimension: int = None):
        """
        Initialize product embedding store

        Args:
            dimension: Embedding dimension (default from config)
        """
        super().__init__(dimension=dimension)
        self.product_ids: List[str] = []
        self.product_metadata: Dict[str, Dict] = {}

    def add_product_embedding(
        self,
        product_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a product embedding to the index

        Args:
            product_id: Product ID
            embedding: Embedding vector
            metadata: Optional metadata (name, category, price, etc.)
        """
        # Use parent's add method
        self.add_customer_embedding(product_id, embedding, metadata)

        # Track product-specific data
        self.product_ids.append(product_id)
        if metadata:
            self.product_metadata[product_id] = metadata

    def search_similar_products(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar products

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            category_filter: Optional category to filter results

        Returns:
            List of product dictionaries with similarity scores
        """
        # Get similar items using parent method
        results = self.search_similar_customers(query_embedding, top_k=top_k * 2)

        # Build product results with metadata
        products = []
        for product_id, similarity in results:
            metadata = self.product_metadata.get(product_id, {})

            # Apply category filter if specified
            if category_filter and metadata.get('category') != category_filter:
                continue

            products.append({
                'product_id': product_id,
                'similarity': similarity,
                **metadata
            })

            if len(products) >= top_k:
                break

        return products

    def get_product_metadata(self, product_id: str) -> Optional[Dict]:
        """
        Get metadata for a product

        Args:
            product_id: Product ID

        Returns:
            Metadata dictionary or None
        """
        return self.product_metadata.get(product_id)
