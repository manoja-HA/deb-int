"""Agent implementations"""

# Import our new intent classifier agent
# Note: Other agents are imported on-demand to avoid circular dependencies
from .intent_classifier_agent import (
    IntentClassifierAgent,
    QueryIntent,
    InformationCategory,
    IntentClassificationResult,
)

def __getattr__(name):
    """Lazy import for other agents to avoid circular dependencies"""
    if name == "customer_profiling_agent":
        from .customer_profiling import customer_profiling_agent
        return customer_profiling_agent
    elif name == "similar_customers_agent":
        from .similar_customers import similar_customers_agent
        return similar_customers_agent
    elif name == "review_filtering_agent":
        from .review_filtering import review_filtering_agent
        return review_filtering_agent
    elif name == "recommendation_agent":
        from .recommendation import recommendation_agent
        return recommendation_agent
    elif name == "response_generation_agent":
        from .response_generation import response_generation_agent
        return response_generation_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "customer_profiling_agent",
    "similar_customers_agent",
    "review_filtering_agent",
    "recommendation_agent",
    "response_generation_agent",
    "IntentClassifierAgent",
    "QueryIntent",
    "InformationCategory",
    "IntentClassificationResult",
]
