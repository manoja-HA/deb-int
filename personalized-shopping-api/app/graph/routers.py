"""
Routing logic for conditional edges in the workflow
"""

from typing import Literal
import logging

from ..state import ShoppingAssistantState

logger = logging.getLogger(__name__)

def route_after_profiling(
    state: ShoppingAssistantState
) -> Literal["similar_customers", "error"]:
    """
    Route after customer profiling

    Returns:
        "similar_customers" if profile found, "error" otherwise
    """
    profile = state.get("customer_profile")

    if profile and profile.get("customer_id"):
        logger.info("✓ Customer profile found, proceeding to similar customer search")
        return "similar_customers"
    else:
        logger.error("✗ Customer profile not found, routing to error")
        return "error"

def route_after_similar_customers(
    state: ShoppingAssistantState
) -> Literal["review_filtering", "fallback"]:
    """
    Route after similar customer discovery

    Returns:
        "review_filtering" if similar customers found, "fallback" otherwise
    """
    similar_customers = state.get("similar_customers", [])

    if similar_customers and len(similar_customers) >= 3:
        logger.info(f"✓ Found {len(similar_customers)} similar customers, proceeding to filtering")
        return "review_filtering"
    else:
        logger.warning(f"✗ Only {len(similar_customers)} similar customers found, using fallback")
        return "fallback"

def route_after_filtering(
    state: ShoppingAssistantState
) -> Literal["recommendation", "popular_fallback"]:
    """
    Route after review filtering

    Returns:
        "recommendation" if enough products, "popular_fallback" otherwise
    """
    filtered_products = state.get("filtered_products", [])

    if filtered_products and len(filtered_products) >= 3:
        logger.info(f"✓ {len(filtered_products)} products passed quality filter, generating recommendations")
        return "recommendation"
    else:
        logger.warning(f"✗ Only {len(filtered_products)} products available, using popular fallback")
        return "popular_fallback"

def route_after_recommendation(
    state: ShoppingAssistantState
) -> Literal["response_generation"]:
    """
    Route after recommendation generation

    Always proceed to response generation
    """
    recommendations = state.get("final_recommendations", [])
    logger.info(f"Generated {len(recommendations)} recommendations, proceeding to response")
    return "response_generation"

def should_retry(state: ShoppingAssistantState) -> bool:
    """
    Determine if workflow should retry on error

    Returns:
        True if retry count below max, False otherwise
    """
    retry_count = state.get("retry_count", 0)
    max_retries = 2  # Allow 2 retries

    return retry_count < max_retries
