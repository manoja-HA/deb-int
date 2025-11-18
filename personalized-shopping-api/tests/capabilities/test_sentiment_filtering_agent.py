"""
Unit tests for SentimentFilteringAgent

These tests demonstrate:
- Testing agents with fake repositories
- Using rule-based sentiment analyzer (no LLM)
- Validating sentiment filtering logic
"""

import pytest
from typing import List, Dict, Any
from app.capabilities.base import AgentContext
from app.capabilities.agents.sentiment_filtering import (
    SentimentFilteringAgent,
    SentimentFilteringInput,
    ProductCandidate,
)
from app.models.sentiment_analyzer import SentimentAnalyzer


class FakeReviewRepository:
    """In-memory review repository for testing"""

    def __init__(self, reviews: Dict[str, List[Dict]]):
        """
        Args:
            reviews: Map of product_id -> list of review dicts
        """
        self.reviews = reviews

    def get_by_product_id(self, product_id: str) -> List[Dict[str, Any]]:
        """Get reviews for a product"""
        return self.reviews.get(product_id, [])


@pytest.fixture
def agent_context():
    """Sample agent context"""
    return AgentContext(
        request_id="test-123",
        session_id="session-456",
        user_id="test-user",
    )


@pytest.fixture
def sentiment_analyzer():
    """Rule-based sentiment analyzer (no LLM)"""
    return SentimentAnalyzer(method="rule_based")


@pytest.fixture
def positive_product_reviews():
    """Product with clearly positive reviews"""
    return {
        "p1": [
            {"review_text": "This product is amazing and wonderful!"},
            {"review_text": "Excellent quality, highly recommend!"},
            {"review_text": "Great purchase, very satisfied!"},
        ]
    }


@pytest.fixture
def negative_product_reviews():
    """Product with clearly negative reviews"""
    return {
        "p2": [
            {"review_text": "Terrible product, very disappointed."},
            {"review_text": "Poor quality, waste of money."},
            {"review_text": "Awful experience, do not buy!"},
        ]
    }


@pytest.fixture
def mixed_product_reviews():
    """Products with positive, negative, and no reviews"""
    return {
        "p1": [  # Positive
            {"review_text": "Excellent product!"},
            {"review_text": "Great quality!"},
        ],
        "p2": [  # Negative
            {"review_text": "Terrible and awful!"},
            {"review_text": "Poor quality!"},
        ],
        "p3": [],  # No reviews
    }


@pytest.mark.asyncio
async def test_filter_positive_product(
    positive_product_reviews,
    sentiment_analyzer,
    agent_context
):
    """Test that product with positive reviews passes filter"""
    repo = FakeReviewRepository(positive_product_reviews)
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p1",
                product_name="Great Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=10,
            )
        ],
        sentiment_threshold=0.6,
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    # Assertions
    assert len(output.filtered_products) == 1
    assert output.products_filtered_out == 0
    assert output.filtered_products[0].product_id == "p1"
    assert output.filtered_products[0].avg_sentiment > 0.6
    assert output.filtered_products[0].review_count == 3


@pytest.mark.asyncio
async def test_filter_negative_product(
    negative_product_reviews,
    sentiment_analyzer,
    agent_context
):
    """Test that product with negative reviews is filtered out"""
    repo = FakeReviewRepository(negative_product_reviews)
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p2",
                product_name="Bad Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=10,
            )
        ],
        sentiment_threshold=0.6,
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    # Assertions
    assert len(output.filtered_products) == 0
    assert output.products_filtered_out == 1


@pytest.mark.asyncio
async def test_product_with_no_reviews(
    mixed_product_reviews,
    sentiment_analyzer,
    agent_context
):
    """Test that product with no reviews gets neutral sentiment and passes"""
    repo = FakeReviewRepository(mixed_product_reviews)
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p3",
                product_name="Unknown Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=5,
            )
        ],
        sentiment_threshold=0.6,
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    # Product with no reviews gets neutral sentiment (0.5) which fails threshold of 0.6
    # So it should be filtered out
    assert len(output.filtered_products) == 0
    assert output.products_filtered_out == 1


@pytest.mark.asyncio
async def test_mixed_products_filtering(
    mixed_product_reviews,
    sentiment_analyzer,
    agent_context
):
    """Test filtering multiple products with mixed sentiments"""
    repo = FakeReviewRepository(mixed_product_reviews)
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p1",
                product_name="Good Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=10,
            ),
            ProductCandidate(
                product_id="p2",
                product_name="Bad Product",
                product_category="Books",
                avg_price=20.0,
                purchase_count=5,
            ),
            ProductCandidate(
                product_id="p3",
                product_name="Unknown Product",
                product_category="Clothing",
                avg_price=50.0,
                purchase_count=3,
            ),
        ],
        sentiment_threshold=0.6,
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    # Only p1 (positive) should pass
    assert len(output.filtered_products) == 1
    assert output.products_filtered_out == 2
    assert output.filtered_products[0].product_id == "p1"
    assert output.products_considered == 3


@pytest.mark.asyncio
async def test_sentiment_threshold_adjustment(
    mixed_product_reviews,
    sentiment_analyzer,
    agent_context
):
    """Test that lowering sentiment threshold allows more products through"""
    repo = FakeReviewRepository(mixed_product_reviews)
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    # Lower threshold to 0.3
    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p1",
                product_name="Good Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=10,
            ),
            ProductCandidate(
                product_id="p3",
                product_name="Unknown Product",
                product_category="Clothing",
                avg_price=50.0,
                purchase_count=3,
            ),
        ],
        sentiment_threshold=0.3,  # Lower threshold
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    # Both should pass with lower threshold
    # p1 has positive reviews (>0.6), p3 has no reviews (0.5)
    assert len(output.filtered_products) == 2
    assert output.products_filtered_out == 0


@pytest.mark.asyncio
async def test_empty_product_list(sentiment_analyzer, agent_context):
    """Test handling of empty product list"""
    repo = FakeReviewRepository({})
    agent = SentimentFilteringAgent(repo, sentiment_analyzer)

    input_data = SentimentFilteringInput(
        candidate_products=[],
        sentiment_threshold=0.6,
        min_reviews=1,
    )

    output = await agent.run(input_data, agent_context)

    assert len(output.filtered_products) == 0
    assert output.products_filtered_out == 0
    assert output.products_considered == 0


@pytest.mark.asyncio
async def test_min_reviews_parameter():
    """Test that min_reviews parameter is passed to sentiment analyzer"""
    reviews = {
        "p1": [
            {"review_text": "Great product!"},
            {"review_text": "Excellent!"},
            {"review_text": "Amazing!"},
            {"review_text": "Wonderful!"},
            {"review_text": "Fantastic!"},
        ]
    }
    repo = FakeReviewRepository(reviews)
    analyzer = SentimentAnalyzer(method="rule_based")
    agent = SentimentFilteringAgent(repo, analyzer)
    context = AgentContext(request_id="test", user_id="test")

    input_data = SentimentFilteringInput(
        candidate_products=[
            ProductCandidate(
                product_id="p1",
                product_name="Product",
                product_category="Electronics",
                avg_price=100.0,
                purchase_count=10,
            )
        ],
        sentiment_threshold=0.5,
        min_reviews=3,  # Require at least 3 reviews
    )

    output = await agent.run(input_data, context)

    # Should pass since it has 5 reviews (> min_reviews=3)
    assert len(output.filtered_products) == 1
    assert output.filtered_products[0].review_count == 5
