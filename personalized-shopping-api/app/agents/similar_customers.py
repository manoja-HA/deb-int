"""
Agent 2: Similar Customer Discovery Agent
Find customers with similar purchase behaviors using vector embeddings
"""

from typing import Dict, List
import logging

from ..state import ShoppingAssistantState, SimilarCustomer
from ..data.loaders import load_purchase_data
from ..data.processors import calculate_category_overlap
from ..models.embedding_model import get_embedding_model
from ..data.embeddings_generator import EmbeddingsGenerator
from ..vector_store import CustomerEmbeddingStore
from ..config import config
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)

@track_agent_performance("similar_customers")
def similar_customers_agent(state: ShoppingAssistantState) -> Dict:
    """
    Agent 2: Find similar customers using vector similarity search

    Steps:
    1. Generate embedding for target customer's purchase pattern
    2. Search vector database for similar customer embeddings
    3. Retrieve top-K similar customers with scores
    4. Load their purchase histories
    5. Return similar customers with metadata

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields
    """
    profile = state.get("customer_profile")

    if not profile:
        logger.error("No customer profile found in state")
        return {
            "errors": ["Cannot find similar customers without profile"],
            "similar_customers": [],
            "agent_execution_order": ["similar_customers"]
        }

    logger.info(f"Finding similar customers for {profile['customer_name']}")

    try:
        # Step 1: Generate embedding for target customer
        embedding_model = get_embedding_model()
        embeddings_gen = EmbeddingsGenerator(embedding_model.model)

        target_embedding = embeddings_gen.generate_customer_embedding(
            customer_id=profile['customer_id'],
            profile=profile,
            use_cache=config.cache_enabled
        )

        # Step 2: Load or create vector store
        vector_store = CustomerEmbeddingStore()

        # Try to load existing index
        try:
            vector_store.load()
            logger.info(f"Loaded vector store with {vector_store.get_index_size()} customers")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            logger.warning("Vector store needs to be built. Run scripts/build_vector_index.py")
            # Return empty results with warning
            return {
                "warnings": ["Vector store not initialized. Please run build_vector_index.py"],
                "similar_customers": [],
                "embedding_model_used": config.embedding_model,
                "agent_execution_order": ["similar_customers"]
            }

        # Step 3: Search for similar customers
        similar_results = vector_store.search_similar_customers(
            query_embedding=target_embedding,
            top_k=config.similarity_top_k,
            threshold=config.similarity_threshold
        )

        # Step 4: Load purchase histories and build results
        similar_customers: List[SimilarCustomer] = []

        for customer_id, similarity_score in similar_results:
            # Skip the target customer themselves
            if customer_id == profile['customer_id']:
                continue

            # Load their purchases
            their_purchases = load_purchase_data(
                customer_id=customer_id,
                limit_days=config.profile_lookback_days
            )

            if not their_purchases:
                continue

            # Get metadata
            metadata = vector_store.get_customer_metadata(customer_id) or {}

            # Calculate common categories
            their_categories = list(set(p['product_category'] for p in their_purchases))
            _, common_categories = calculate_category_overlap(
                profile['favorite_categories'],
                their_categories
            )

            similar_customer = SimilarCustomer(
                customer_id=customer_id,
                customer_name=metadata.get('customer_name', f"Customer_{customer_id}"),
                similarity_score=similarity_score,
                common_categories=common_categories,
                their_purchases=their_purchases
            )

            similar_customers.append(similar_customer)

        logger.info(
            f"Found {len(similar_customers)} similar customers "
            f"(threshold: {config.similarity_threshold})"
        )

        # Log top matches
        if similar_customers:
            top_3 = similar_customers[:3]
            for sc in top_3:
                logger.info(
                    f"  - {sc['customer_name']}: {sc['similarity_score']:.3f} "
                    f"(common: {', '.join(sc['common_categories'])})"
                )

        return {
            "similar_customers": similar_customers,
            "embedding_model_used": config.embedding_model,
            "agent_execution_order": ["similar_customers"]
        }

    except Exception as e:
        logger.error(f"Similar customer search failed: {e}", exc_info=True)
        return {
            "errors": [f"Similar customer search error: {str(e)}"],
            "similar_customers": [],
            "fallback_used": True,
            "agent_execution_order": ["similar_customers"]
        }
