"""
Capabilities package - Reusable agent components

This package contains the base agent architecture and concrete agent implementations
that follow a uniform interface for composability and testability.
"""

from app.capabilities.base import (
    AgentContext,
    BaseAgent,
    AgentMetadata,
    AgentRegistry,
)

__all__ = [
    "AgentContext",
    "BaseAgent",
    "AgentMetadata",
    "AgentRegistry",
    "register_all_agents",
]


def register_all_agents() -> None:
    """
    Register all built-in production agents with the AgentRegistry.

    This function should be called during application startup to make
    agents discoverable for runtime introspection and tooling.

    Registered agents:
    - CustomerProfilingAgent
    - SimilarCustomersAgent
    - SentimentFilteringAgent
    - ProductScoringAgent
    - ResponseGenerationAgent
    - ResponseGenerationAgentV2
    """
    from app.capabilities.agents.customer_profiling import CustomerProfilingAgent
    from app.capabilities.agents.similar_customers import SimilarCustomersAgent
    from app.capabilities.agents.sentiment_filtering import SentimentFilteringAgent
    from app.capabilities.agents.product_scoring import ProductScoringAgent
    from app.capabilities.agents.response_generation import ResponseGenerationAgent
    from app.capabilities.agents.response_generation_v2 import ResponseGenerationAgentV2

    # Register each agent with its metadata
    # Note: We register the agent class and metadata for discovery purposes

    # CustomerProfilingAgent
    AgentRegistry.register(
        CustomerProfilingAgent,
        AgentMetadata(
            id="customer_profiling",
            name="Customer Profiling Agent",
            description="Extracts behavioral metrics and segments from customer purchase history",
            version="1.0.0",
            tags=["customer", "profiling", "segmentation"],
        ),
    )

    # SimilarCustomersAgent
    AgentRegistry.register(
        SimilarCustomersAgent,
        AgentMetadata(
            id="similar_customers",
            name="Similar Customers Agent",
            description="Discovers customers with similar behavior using vector similarity search",
            version="1.0.0",
            tags=["similarity", "vector", "collaborative"],
        ),
    )

    # SentimentFilteringAgent
    AgentRegistry.register(
        SentimentFilteringAgent,
        AgentMetadata(
            id="sentiment_filtering",
            name="Sentiment Filtering Agent",
            description="Filters products based on review sentiment analysis",
            version="1.0.0",
            tags=["filtering", "sentiment", "reviews", "quality"],
        ),
    )

    # ProductScoringAgent
    AgentRegistry.register(
        ProductScoringAgent,
        AgentMetadata(
            id="product_scoring",
            name="Product Scoring Agent",
            description="Scores and ranks products using collaborative filtering and category affinity",
            version="1.0.0",
            tags=["scoring", "ranking", "collaborative", "recommendation"],
        ),
    )

    # ResponseGenerationAgent (original)
    AgentRegistry.register(
        ResponseGenerationAgent,
        AgentMetadata(
            id="response_generation",
            name="Response Generation Agent",
            description="Generates natural language explanations for recommendations using LLM",
            version="1.0.0",
            tags=["response", "llm", "reasoning", "explanation"],
        ),
    )

    # ResponseGenerationAgentV2 (PydanticAI version)
    AgentRegistry.register(
        ResponseGenerationAgentV2,
        AgentMetadata(
            id="response_generation_v2",
            name="Response Generation Agent V2",
            description="PydanticAI-based response generation with structured outputs",
            version="2.0.0",
            tags=["response", "llm", "reasoning", "pydantic-ai"],
        ),
    )
