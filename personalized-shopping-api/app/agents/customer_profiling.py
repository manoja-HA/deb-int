"""
Agent 1: Customer Profiling Agent
Extract customer profile from purchase history
"""

from typing import Dict
import pandas as pd
import logging

from ..state import ShoppingAssistantState, CustomerProfile
from ..data.loaders import load_purchase_data, get_customer_id_by_name
from ..data.processors import calculate_customer_metrics
from ..config import config
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)

@track_agent_performance("customer_profiling")
def customer_profiling_agent(state: ShoppingAssistantState) -> Dict:
    """
    Agent 1: Extract customer profile from purchase history

    Steps:
    1. Lookup customer ID if name provided
    2. Load purchase history
    3. Calculate profile metrics
    4. Determine customer segment
    5. Build CustomerProfile object
    6. Calculate confidence score

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields
    """
    customer_name = state.get("customer_name")
    customer_id = state.get("customer_id")

    logger.info(f"Profiling customer: {customer_name or customer_id}")

    try:
        # Step 1: Resolve customer ID
        if not customer_id and customer_name:
            customer_id = get_customer_id_by_name(customer_name)
            if not customer_id:
                logger.error(f"Customer not found: {customer_name}")
                return {
                    "errors": [f"Customer not found: {customer_name}"],
                    "customer_profile": None,
                    "confidence_score": 0.0,
                    "agent_execution_order": ["customer_profiling"]
                }

        # Step 2: Load purchase history
        purchases = load_purchase_data(
            customer_id=customer_id,
            limit_days=config.profile_lookback_days
        )

        if not purchases:
            logger.error(f"No purchase history found for customer {customer_id}")
            return {
                "errors": [f"No purchase history found for customer {customer_id}"],
                "customer_profile": None,
                "confidence_score": 0.0,
                "agent_execution_order": ["customer_profiling"]
            }

        # Step 3: Calculate metrics
        metrics = calculate_customer_metrics(purchases)

        # Step 4: Extract favorite brands (if available in data)
        df = pd.DataFrame(purchases)
        favorite_brands = []  # Placeholder - implement if brand data available

        # Step 5: Build profile
        profile = CustomerProfile(
            customer_id=customer_id,
            customer_name=customer_name or f"Customer_{customer_id}",
            total_purchases=metrics['total_purchases'],
            total_spent=metrics['total_spent'],
            avg_purchase_price=metrics['avg_purchase_price'],
            favorite_categories=metrics['favorite_categories'],
            favorite_brands=favorite_brands,
            purchase_frequency=metrics['purchase_frequency'],
            price_segment=metrics['price_segment'],
            recent_purchases=purchases[-5:],  # Last 5 purchases
            confidence=min(metrics['total_purchases'] / 10.0, 1.0)
        )

        # Step 6: Log profile summary
        logger.info(
            f"Profile created: {profile['total_purchases']} purchases, "
            f"{profile['purchase_frequency']} frequency, "
            f"{profile['price_segment']} segment, "
            f"categories: {', '.join(profile['favorite_categories'])}"
        )

        return {
            "customer_profile": profile,
            "customer_id": customer_id,
            "profiling_method": "purchase_history_analysis",
            "agent_execution_order": ["customer_profiling"],
            "confidence_score": profile['confidence']
        }

    except Exception as e:
        logger.error(f"Customer profiling failed: {e}", exc_info=True)
        return {
            "errors": [f"Profiling error: {str(e)}"],
            "customer_profile": None,
            "fallback_used": True,
            "agent_execution_order": ["customer_profiling"]
        }
