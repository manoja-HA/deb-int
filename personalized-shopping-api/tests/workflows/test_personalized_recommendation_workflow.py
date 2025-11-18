"""
Integration tests for PersonalizedRecommendationWorkflow

These tests demonstrate:
- End-to-end workflow testing with stub repositories
- No network calls or LLM invocations
- Fast, deterministic test execution
"""

import pytest
from typing import List, Dict, Any
import numpy as np
from app.capabilities.base import AgentContext
from app.workflows.personalized_recommendation import PersonalizedRecommendationWorkflow
from app.models.sentiment_analyzer import SentimentAnalyzer


class StubCustomerRepository:
    """Stub customer repository with deterministic in-memory data"""

    def __init__(self):
        self.customers = {
            "cust1": {"customer_id": "cust1", "customer_name": "Alice"},
            "cust2": {"customer_id": "cust2", "customer_name": "Bob"},
            "cust3": {"customer_id": "cust3", "customer_name": "Charlie"},
        }
        self.purchases = {
            "cust1": [
                {
                    "transaction_id": "t1",
                    "product_id": "p1",
                    "product_name": "Laptop",
                    "product_category": "Electronics",
                    "price": 800.0,
                },
                {
                    "transaction_id": "t2",
                    "product_id": "p2",
                    "product_name": "Mouse",
                    "product_category": "Electronics",
                    "price": 25.0,
                },
            ],
            "cust2": [
                {
                    "transaction_id": "t3",
                    "product_id": "p3",
                    "product_name": "Keyboard",
                    "product_category": "Electronics",
                    "price": 60.0,
                },
                {
                    "transaction_id": "t4",
                    "product_id": "p4",
                    "product_name": "Monitor",
                    "product_category": "Electronics",
                    "price": 300.0,
                },
            ],
            "cust3": [
                {
                    "transaction_id": "t5",
                    "product_id": "p5",
                    "product_name": "Headphones",
                    "product_category": "Electronics",
                    "price": 150.0,
                },
            ],
        }

    def get_by_id(self, customer_id: str) -> Dict[str, Any] | None:
        return self.customers.get(customer_id)

    def get_purchases_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        return self.purchases.get(customer_id, [])


class StubVectorRepository:
    """Stub vector repository returning similar customers"""

    def __init__(self):
        # Predefined similarity results
        self.similar_customers = {
            "cust1": ["cust2", "cust3"],  # Alice is similar to Bob and Charlie
            "cust2": ["cust1", "cust3"],
            "cust3": ["cust1", "cust2"],
        }

    def search_similar_customers(
        self,
        query_text: str,
        top_k: int = 20,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Return stub similar customer results"""
        # Extract customer_id from query_text if possible
        # For testing, we'll return all similar customers
        return [
            {"customer_id": "cust2", "similarity": 0.9},
            {"customer_id": "cust3", "similarity": 0.85},
        ]


class StubReviewRepository:
    """Stub review repository with positive reviews for all products"""

    def __init__(self):
        self.reviews = {
            "p3": [
                {"review_text": "Excellent keyboard!"},
                {"review_text": "Great quality!"},
            ],
            "p4": [
                {"review_text": "Amazing monitor!"},
                {"review_text": "Perfect display!"},
            ],
            "p5": [
                {"review_text": "Wonderful headphones!"},
                {"review_text": "Great sound quality!"},
            ],
        }

    def get_by_product_id(self, product_id: str) -> List[Dict[str, Any]]:
        return self.reviews.get(product_id, [])


@pytest.fixture
def stub_repositories():
    """Create stub repositories"""
    return {
        "customer_repo": StubCustomerRepository(),
        "vector_repo": StubVectorRepository(),
        "review_repo": StubReviewRepository(),
        "sentiment_analyzer": SentimentAnalyzer(method="rule_based"),
    }


@pytest.fixture
def agent_context():
    """Sample agent context"""
    return AgentContext(
        request_id="test-workflow-123",
        session_id="session-456",
        user_id="cust1",
    )


@pytest.mark.asyncio
async def test_workflow_end_to_end(stub_repositories, agent_context):
    """Test complete workflow execution with stub data"""
    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=stub_repositories["customer_repo"],
        vector_repository=stub_repositories["vector_repo"],
        review_repository=stub_repositories["review_repo"],
        sentiment_analyzer=stub_repositories["sentiment_analyzer"],
    )

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products for Alice",
        top_n=5,
        include_reasoning=False,  # Skip LLM reasoning for speed
        context=agent_context,
    )

    # Assertions
    assert response is not None
    assert response.customer_profile.customer_id == "cust1"
    assert response.customer_profile.customer_name == "Alice"
    assert response.customer_profile.total_purchases == 2
    assert "Electronics" in response.customer_profile.favorite_categories

    # Should have recommendations from similar customers
    assert len(response.recommendations) > 0
    assert response.similar_customers_analyzed >= 0
    assert response.products_considered >= 0

    # Agent execution order should be present
    assert "customer_profiling" in response.agent_execution_order
    assert "similar_customer_discovery" in response.agent_execution_order
    assert "sentiment_filtering" in response.agent_execution_order
    assert "product_scoring" in response.agent_execution_order

    # Processing time should be reasonable
    assert response.processing_time_ms > 0


@pytest.mark.asyncio
async def test_workflow_recommendation_quality(stub_repositories):
    """Test that recommendations are properly scored and ranked"""
    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=stub_repositories["customer_repo"],
        vector_repository=stub_repositories["vector_repo"],
        review_repository=stub_repositories["review_repo"],
        sentiment_analyzer=stub_repositories["sentiment_analyzer"],
    )

    context = AgentContext(request_id="test-quality", user_id="cust1")

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products",
        top_n=3,
        include_reasoning=False,
        context=context,
    )

    # Recommendations should be sorted by score (descending)
    if len(response.recommendations) > 1:
        scores = [rec.recommendation_score for rec in response.recommendations]
        assert scores == sorted(scores, reverse=True)

    # Each recommendation should have required fields
    for rec in response.recommendations:
        assert rec.product_id
        assert rec.product_name
        assert rec.product_category == "Electronics"
        assert rec.recommendation_score >= 0
        assert rec.recommendation_score <= 1
        assert rec.reason  # Should have reasoning
        assert rec.source in ["collaborative", "category_affinity", "trending"]


@pytest.mark.asyncio
async def test_workflow_excludes_already_purchased(stub_repositories):
    """Test that workflow excludes products customer already bought"""
    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=stub_repositories["customer_repo"],
        vector_repository=stub_repositories["vector_repo"],
        review_repository=stub_repositories["review_repo"],
        sentiment_analyzer=stub_repositories["sentiment_analyzer"],
    )

    context = AgentContext(request_id="test-exclusion", user_id="cust1")

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products",
        top_n=5,
        include_reasoning=False,
        context=context,
    )

    # cust1 already bought p1 (Laptop) and p2 (Mouse)
    # These should NOT appear in recommendations
    recommended_ids = [rec.product_id for rec in response.recommendations]
    assert "p1" not in recommended_ids
    assert "p2" not in recommended_ids


@pytest.mark.asyncio
async def test_workflow_empty_recommendations():
    """Test workflow behavior when no recommendations can be made"""
    # Create repos with minimal data
    customer_repo = StubCustomerRepository()

    # Vector repo that returns no similar customers
    class EmptyVectorRepo:
        def search_similar_customers(self, query_text: str, top_k: int = 20, similarity_threshold: float = 0.75):
            return []

    review_repo = StubReviewRepository()
    sentiment_analyzer = SentimentAnalyzer(method="rule_based")

    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=customer_repo,
        vector_repository=EmptyVectorRepo(),
        review_repository=review_repo,
        sentiment_analyzer=sentiment_analyzer,
    )

    context = AgentContext(request_id="test-empty", user_id="cust1")

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products",
        top_n=5,
        include_reasoning=False,
        context=context,
    )

    # Should return empty recommendations gracefully
    assert len(response.recommendations) == 0
    assert response.confidence_score == 0.0
    assert "no suitable recommendations" in response.reasoning.lower()


@pytest.mark.asyncio
async def test_workflow_context_metadata(stub_repositories):
    """Test that workflow properly tracks execution metadata"""
    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=stub_repositories["customer_repo"],
        vector_repository=stub_repositories["vector_repo"],
        review_repository=stub_repositories["review_repo"],
        sentiment_analyzer=stub_repositories["sentiment_analyzer"],
    )

    context = AgentContext(
        request_id="test-metadata",
        session_id="session-123",
        user_id="cust1",
        metadata={"source": "test"},
    )

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products",
        top_n=5,
        include_reasoning=False,
        context=context,
    )

    # Check metadata is preserved
    assert response.metadata["request_id"] == "test-metadata"
    assert response.metadata["session_id"] == "session-123"
    assert response.metadata["fallback_used"] == False


@pytest.mark.asyncio
async def test_workflow_diversity_constraint(stub_repositories):
    """Test that recommendations respect diversity constraints"""
    workflow = PersonalizedRecommendationWorkflow(
        customer_repository=stub_repositories["customer_repo"],
        vector_repository=stub_repositories["vector_repo"],
        review_repository=stub_repositories["review_repo"],
        sentiment_analyzer=stub_repositories["sentiment_analyzer"],
    )

    context = AgentContext(request_id="test-diversity", user_id="cust1")

    response = await workflow.execute(
        customer_id="cust1",
        query="Recommend products",
        top_n=5,
        include_reasoning=False,
        context=context,
    )

    # Count products per category
    category_counts = {}
    for rec in response.recommendations:
        category = rec.product_category
        category_counts[category] = category_counts.get(category, 0) + 1

    # Each category should have at most 2 products (max_per_category=2 in workflow)
    for category, count in category_counts.items():
        assert count <= 2, f"Category '{category}' has {count} products, max is 2"
