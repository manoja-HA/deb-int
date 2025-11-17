"""
Agent 3: Review-Based Filtering Agent
Filter product recommendations to only high-quality items based on sentiment
"""

from typing import Dict, List
from collections import defaultdict
import logging

from ..state import ShoppingAssistantState
from ..data.loaders import load_review_data
from ..data.processors import aggregate_products_from_purchases
from ..models.sentiment_analyzer import SentimentAnalyzer
from ..config import config
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)

@track_agent_performance("review_filtering")
def review_filtering_agent(state: ShoppingAssistantState) -> Dict:
    """
    Agent 3: Filter products based on review sentiment

    Steps:
    1. Extract unique product IDs from similar customers' purchases
    2. Load reviews for these products
    3. Run sentiment analysis on review text
    4. Calculate average sentiment per product
    5. Filter products below sentiment threshold
    6. Return filtered product list

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields
    """
    similar_customers = state.get("similar_customers", [])
    profile = state.get("customer_profile")

    if not similar_customers:
        logger.warning("No similar customers found for review filtering")
        return {
            "warnings": ["No similar customers to extract product recommendations"],
            "candidate_products": [],
            "filtered_products": [],
            "products_filtered_out": 0,
            "agent_execution_order": ["review_filtering"]
        }

    logger.info(f"Filtering products from {len(similar_customers)} similar customers")

    try:
        # Step 1: Aggregate all purchases from similar customers
        all_purchases = []
        for similar_customer in similar_customers:
            all_purchases.extend(similar_customer['their_purchases'])

        # Exclude products already purchased by target customer
        target_product_ids = set(
            p['product_id'] for p in profile.get('recent_purchases', [])
        )

        candidate_purchases = [
            p for p in all_purchases
            if p['product_id'] not in target_product_ids
        ]

        # Aggregate to unique products
        candidate_products = aggregate_products_from_purchases(candidate_purchases)

        logger.info(f"Found {len(candidate_products)} candidate products")

        # Step 2 & 3: Load reviews and analyze sentiment
        sentiment_analyzer = SentimentAnalyzer(method="llm")
        product_sentiments = {}

        for product in candidate_products:
            product_id = product['product_id']

            # Load reviews for this product
            reviews = load_review_data(product_id=product_id)

            if not reviews:
                # No reviews - use purchase frequency as proxy
                logger.debug(f"No reviews for product {product_id}, using neutral score")
                product_sentiments[product_id] = {
                    'avg_sentiment': 0.5,  # Neutral
                    'confidence': 0.3,  # Low confidence
                    'review_count': 0
                }
                continue

            # Analyze sentiment
            review_texts = [r['review_text'] for r in reviews]
            sentiment_result = sentiment_analyzer.calculate_average_sentiment(
                review_texts,
                min_reviews=config.min_reviews_for_inclusion
            )

            product_sentiments[product_id] = sentiment_result

        # Step 4 & 5: Filter by sentiment threshold
        filtered_products = []
        filtered_out_count = 0

        for product in candidate_products:
            product_id = product['product_id']
            sentiment_data = product_sentiments.get(product_id, {})

            avg_sentiment = sentiment_data.get('avg_sentiment', 0.5)
            confidence = sentiment_data.get('confidence', 0.0)
            review_count = sentiment_data.get('review_count', 0)

            # Apply threshold
            if avg_sentiment >= config.sentiment_threshold:
                # Add sentiment data to product
                product['avg_sentiment'] = avg_sentiment
                product['sentiment_confidence'] = confidence
                product['review_count'] = review_count
                filtered_products.append(product)
            else:
                filtered_out_count += 1
                logger.debug(
                    f"Filtered out {product['product_name']}: "
                    f"sentiment {avg_sentiment:.2f} < {config.sentiment_threshold}"
                )

        logger.info(
            f"Filtered to {len(filtered_products)} high-quality products "
            f"(removed {filtered_out_count})"
        )

        # Log top products by sentiment
        if filtered_products:
            sorted_products = sorted(
                filtered_products,
                key=lambda x: x.get('avg_sentiment', 0),
                reverse=True
            )[:3]

            for prod in sorted_products:
                logger.info(
                    f"  - {prod['product_name']}: "
                    f"sentiment {prod['avg_sentiment']:.2f} "
                    f"({prod['review_count']} reviews)"
                )

        return {
            "candidate_products": candidate_products,
            "filtered_products": filtered_products,
            "products_filtered_out": filtered_out_count,
            "agent_execution_order": ["review_filtering"]
        }

    except Exception as e:
        logger.error(f"Review filtering failed: {e}", exc_info=True)
        return {
            "errors": [f"Review filtering error: {str(e)}"],
            "candidate_products": [],
            "filtered_products": [],
            "fallback_used": True,
            "agent_execution_order": ["review_filtering"]
        }
