"""
Unit tests for ProductScoringAgent

These tests demonstrate:
- Testing agents in isolation with no network/database dependencies
- Using in-memory data and mock objects
- Validating agent logic independently
"""

import pytest
from app.capabilities.base import AgentContext
from app.capabilities.agents.product_scoring import (
    ProductScoringAgent,
    ProductScoringInput,
    ScoredProduct,
)
from app.domain.schemas.customer import CustomerProfile


@pytest.fixture
def sample_customer_profile():
    """Sample customer profile for testing"""
    return CustomerProfile(
        customer_id="123",
        customer_name="Test Customer",
        total_purchases=10,
        total_spent=5000.0,
        avg_purchase_price=500.0,
        favorite_categories=["Electronics", "Books", "Clothing"],
        favorite_brands=[],
        price_segment="mid-range",
        purchase_frequency="high",
        recent_purchases=[],
        confidence=1.0,
    )


@pytest.fixture
def sample_products():
    """Sample products for testing"""
    return [
        ScoredProduct(
            product_id="1",
            product_name="Laptop",
            product_category="Electronics",
            avg_price=800.0,
            purchase_count=10,
            avg_sentiment=0.9,
            review_count=50,
        ),
        ScoredProduct(
            product_id="2",
            product_name="Python Book",
            product_category="Books",
            avg_price=50.0,
            purchase_count=8,
            avg_sentiment=0.85,
            review_count=30,
        ),
        ScoredProduct(
            product_id="3",
            product_name="T-Shirt",
            product_category="Clothing",
            avg_price=25.0,
            purchase_count=5,
            avg_sentiment=0.7,
            review_count=20,
        ),
        ScoredProduct(
            product_id="4",
            product_name="Mouse",
            product_category="Electronics",
            avg_price=30.0,
            purchase_count=3,
            avg_sentiment=0.8,
            review_count=15,
        ),
    ]


@pytest.fixture
def agent_context():
    """Sample agent context"""
    return AgentContext(
        request_id="test-123",
        session_id="session-456",
        user_id="test-user",
    )


@pytest.mark.asyncio
async def test_product_scoring_basic(
    sample_customer_profile,
    sample_products,
    agent_context
):
    """Test basic product scoring and ranking"""
    agent = ProductScoringAgent()

    # Create purchase counts
    purchase_counts = {
        "1": 10,  # Laptop - most popular
        "2": 8,   # Python Book
        "3": 5,   # T-Shirt
        "4": 3,   # Mouse
    }

    input_data = ProductScoringInput(
        customer_profile=sample_customer_profile,
        products=sample_products,
        purchase_counts=purchase_counts,
        top_n=3,
        max_per_category=2,
    )

    output = await agent.run(input_data, agent_context)

    # Assertions
    assert len(output.recommendations) == 3
    assert output.total_scored == 4

    # Check that recommendations are sorted by score
    scores = [rec.recommendation_score for rec in output.recommendations]
    assert scores == sorted(scores, reverse=True)

    # Check that each recommendation has required fields
    for rec in output.recommendations:
        assert rec.product_id
        assert rec.product_name
        assert rec.reason
        assert 0 <= rec.recommendation_score <= 1
        assert rec.source in ["collaborative", "category_affinity", "trending"]


@pytest.mark.asyncio
async def test_product_scoring_category_affinity(
    sample_customer_profile,
    agent_context
):
    """Test category affinity scoring"""
    agent = ProductScoringAgent()

    # Products in different categories
    products = [
        ScoredProduct(
            product_id="1",
            product_name="Phone",
            product_category="Electronics",  # #1 favorite
            avg_price=500.0,
            purchase_count=5,
            avg_sentiment=0.8,
            review_count=10,
        ),
        ScoredProduct(
            product_id="2",
            product_name="Novel",
            product_category="Books",  # #2 favorite
            avg_price=20.0,
            purchase_count=5,
            avg_sentiment=0.8,
            review_count=10,
        ),
        ScoredProduct(
            product_id="3",
            product_name="Toy",
            product_category="Toys",  # Not in favorites
            avg_price=30.0,
            purchase_count=5,
            avg_sentiment=0.8,
            review_count=10,
        ),
    ]

    input_data = ProductScoringInput(
        customer_profile=sample_customer_profile,
        products=products,
        purchase_counts={"1": 5, "2": 5, "3": 5},  # Same collaborative score
        top_n=3,
    )

    output = await agent.run(input_data, agent_context)

    # Electronics should rank higher than Books due to category affinity
    # Both should rank higher than Toys
    product_ids = [rec.product_id for rec in output.recommendations]
    assert product_ids.index("1") < product_ids.index("2")
    assert product_ids.index("2") < product_ids.index("3")


@pytest.mark.asyncio
async def test_product_scoring_diversity_constraint(
    sample_customer_profile,
    agent_context
):
    """Test diversity constraint (max 2 per category)"""
    agent = ProductScoringAgent()

    # All products in same category
    products = [
        ScoredProduct(
            product_id=str(i),
            product_name=f"Electronics {i}",
            product_category="Electronics",
            avg_price=100.0 * i,
            purchase_count=10 - i,
            avg_sentiment=0.8,
            review_count=10,
        )
        for i in range(1, 6)
    ]

    purchase_counts = {str(i): 10 - i for i in range(1, 6)}

    input_data = ProductScoringInput(
        customer_profile=sample_customer_profile,
        products=products,
        purchase_counts=purchase_counts,
        top_n=5,
        max_per_category=2,  # Limit to 2 per category
    )

    output = await agent.run(input_data, agent_context)

    # Should only get 2 products despite top_n=5
    assert len(output.recommendations) == 2

    # Both should be Electronics
    for rec in output.recommendations:
        assert rec.product_category == "Electronics"


@pytest.mark.asyncio
async def test_product_scoring_empty_products(
    sample_customer_profile,
    agent_context
):
    """Test handling of empty product list"""
    agent = ProductScoringAgent()

    input_data = ProductScoringInput(
        customer_profile=sample_customer_profile,
        products=[],
        purchase_counts={},
        top_n=5,
    )

    output = await agent.run(input_data, agent_context)

    assert len(output.recommendations) == 0
    assert output.total_scored == 0


@pytest.mark.asyncio
async def test_category_affinity_calculation():
    """Test internal category affinity calculation"""
    agent = ProductScoringAgent()

    target_categories = ["Electronics", "Books", "Clothing"]

    # First favorite category
    assert agent._calculate_category_affinity(target_categories, "Electronics") == 1.0

    # Second favorite category
    assert agent._calculate_category_affinity(target_categories, "Books") == 0.8

    # Third favorite category
    assert agent._calculate_category_affinity(target_categories, "Clothing") == 0.6

    # Not in favorites
    assert agent._calculate_category_affinity(target_categories, "Toys") == 0.2

    # Empty favorites
    assert agent._calculate_category_affinity([], "Electronics") == 0.5
