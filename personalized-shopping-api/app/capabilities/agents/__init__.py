"""
Concrete agent implementations

Each agent is a single-purpose capability that implements the BaseAgent interface.
Agents are composable, testable, and reusable across different workflows.
"""

from app.capabilities.agents.customer_profiling import CustomerProfilingAgent
from app.capabilities.agents.similar_customers import SimilarCustomersAgent
from app.capabilities.agents.sentiment_filtering import SentimentFilteringAgent
from app.capabilities.agents.product_scoring import ProductScoringAgent
from app.capabilities.agents.response_generation import ResponseGenerationAgent

__all__ = [
    "CustomerProfilingAgent",
    "SimilarCustomersAgent",
    "SentimentFilteringAgent",
    "ProductScoringAgent",
    "ResponseGenerationAgent",
]
