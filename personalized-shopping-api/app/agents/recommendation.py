"""
Agent 4: Cross-Category Recommendation Agent
Generate final recommendations with collaborative filtering and category affinity
"""

from typing import Dict, List
import logging
from collections import Counter

from ..state import ShoppingAssistantState, ProductRecommendation
from ..data.processors import calculate_category_affinity, normalize_scores
from ..config import config
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)

@track_agent_performance("recommendation")
def recommendation_agent(state: ShoppingAssistantState) -> Dict:
    """
    Agent 4: Generate final product recommendations

    Steps:
    1. Score filtered products using collaborative filtering
    2. Apply category affinity weighting
    3. Add diversity (max 2 products per category)
    4. Generate explanation for each recommendation
    5. Return top-N with scores and reasoning

    Scoring Formula:
    final_score = (collaborative_weight * collab_score) +
                  (category_weight * category_score) +
                  (sentiment_boost * avg_sentiment)

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields
    """
    filtered_products = state.get("filtered_products", [])
    profile = state.get("customer_profile")
    similar_customers = state.get("similar_customers", [])

    if not filtered_products:
        logger.warning("No filtered products available for recommendation")
        return {
            "warnings": ["No products passed quality filters"],
            "final_recommendations": [],
            "recommendation_strategy": "none",
            "agent_execution_order": ["recommendation"]
        }

    logger.info(f"Generating recommendations from {len(filtered_products)} products")

    try:
        # Step 1: Calculate collaborative filtering score
        # Count how many similar customers bought each product
        product_purchase_counts = Counter()

        for similar_customer in similar_customers:
            for purchase in similar_customer['their_purchases']:
                product_purchase_counts[purchase['product_id']] += 1

        # Calculate collaborative scores (normalized by number of similar customers)
        max_count = max(product_purchase_counts.values()) if product_purchase_counts else 1

        for product in filtered_products:
            product_id = product['product_id']
            purchase_count = product_purchase_counts.get(product_id, 0)
            product['collab_score'] = purchase_count / max_count

        # Step 2: Calculate category affinity scores
        target_categories = profile.get('favorite_categories', [])

        for product in filtered_products:
            category_score = calculate_category_affinity(
                target_categories,
                product['product_category']
            )
            product['category_score'] = category_score

        # Step 3: Calculate final scores
        for product in filtered_products:
            collab_score = product.get('collab_score', 0)
            category_score = product.get('category_score', 0)
            sentiment_score = product.get('avg_sentiment', 0.5)

            # Weighted combination
            final_score = (
                config.collaborative_weight * collab_score +
                config.category_affinity_weight * category_score +
                0.2 * sentiment_score  # Small sentiment boost
            )

            product['final_score'] = final_score

        # Sort by final score
        sorted_products = sorted(
            filtered_products,
            key=lambda x: x['final_score'],
            reverse=True
        )

        # Step 4: Add diversity (max 2 per category)
        category_counts = Counter()
        diverse_products = []

        for product in sorted_products:
            category = product['product_category']

            if category_counts[category] < 2:  # Max 2 per category
                diverse_products.append(product)
                category_counts[category] += 1

            if len(diverse_products) >= config.recommendation_top_n:
                break

        # Step 5: Build recommendation objects with explanations
        recommendations: List[ProductRecommendation] = []

        for product in diverse_products:
            # Determine source/reason
            collab_score = product.get('collab_score', 0)
            category_score = product.get('category_score', 0)

            if collab_score > 0.7:
                source = "collaborative"
                reason = f"Highly popular with {int(product_purchase_counts[product['product_id']])} similar customers"
            elif category_score > 0.8:
                source = "category_affinity"
                reason = f"Matches your preference for {product['product_category']}"
            else:
                source = "trending"
                reason = "Good balance of popularity and category fit"

            # Add quality note
            if product.get('avg_sentiment', 0) > 0.8:
                reason += f" with excellent reviews ({product.get('avg_sentiment', 0):.0%} positive)"

            recommendation = ProductRecommendation(
                product_id=product['product_id'],
                product_name=product['product_name'],
                product_category=product['product_category'],
                avg_price=product['avg_price'],
                recommendation_score=product['final_score'],
                reason=reason,
                similar_customer_count=product_purchase_counts[product['product_id']],
                avg_sentiment=product.get('avg_sentiment', 0.5),
                source=source
            )

            recommendations.append(recommendation)

        logger.info(f"Generated {len(recommendations)} final recommendations")

        # Log recommendations
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"  {i}. {rec['product_name']} (${rec['avg_price']:.0f}): "
                f"score {rec['recommendation_score']:.3f} - {rec['reason']}"
            )

        return {
            "final_recommendations": recommendations,
            "recommendation_strategy": "collaborative_with_category_affinity",
            "agent_execution_order": ["recommendation"]
        }

    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        return {
            "errors": [f"Recommendation error: {str(e)}"],
            "final_recommendations": [],
            "fallback_used": True,
            "agent_execution_order": ["recommendation"]
        }
