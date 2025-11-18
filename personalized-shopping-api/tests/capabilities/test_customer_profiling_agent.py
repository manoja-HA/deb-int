"""
Unit tests for CustomerProfilingAgent

These tests demonstrate:
- Testing agents in isolation with no network/database dependencies
- Using in-memory data and mock objects
- Validating agent logic independently
"""

import pytest
from typing import List, Dict, Any
from app.capabilities.base import AgentContext
from app.capabilities.agents.customer_profiling import (
    CustomerProfilingAgent,
    CustomerProfilingInput,
)


class FakeCustomerRepository:
    """In-memory customer repository for testing"""

    def __init__(self, customers: List[Dict], purchases: Dict[str, List[Dict]]):
        """
        Args:
            customers: List of customer dicts
            purchases: Map of customer_id -> list of purchase dicts
        """
        self.customers = {c["customer_id"]: c for c in customers}
        self.purchases = purchases

    def get_by_id(self, customer_id: str) -> Dict[str, Any] | None:
        """Get customer by ID"""
        return self.customers.get(customer_id)

    def get_purchases_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get purchases for a customer"""
        return self.purchases.get(customer_id, [])


@pytest.fixture
def agent_context():
    """Sample agent context"""
    return AgentContext(
        request_id="test-123",
        session_id="session-456",
        user_id="test-user",
    )


@pytest.fixture
def budget_customer_repo():
    """Customer with budget segment behavior"""
    customers = [
        {"customer_id": "1", "customer_name": "Budget Bob"}
    ]
    purchases = {
        "1": [
            {
                "transaction_id": f"t{i}",
                "product_id": f"p{i}",
                "product_name": f"Cheap Product {i}",
                "product_category": "Electronics" if i < 2 else "Books",
                "price": 50.0 + i * 10,
            }
            for i in range(3)  # 3 purchases, avg ~$70
        ]
    }
    return FakeCustomerRepository(customers, purchases)


@pytest.fixture
def premium_customer_repo():
    """Customer with premium segment behavior"""
    customers = [
        {"customer_id": "2", "customer_name": "Premium Petra"}
    ]
    purchases = {
        "2": [
            {
                "transaction_id": f"t{i}",
                "product_id": f"p{i}",
                "product_name": f"Luxury Product {i}",
                "product_category": "Electronics" if i < 8 else "Clothing",
                "price": 700.0 + i * 50,
            }
            for i in range(12)  # 12 purchases, avg ~$975
        ]
    }
    return FakeCustomerRepository(customers, purchases)


@pytest.mark.asyncio
async def test_budget_customer_profiling(budget_customer_repo, agent_context):
    """Test profiling a budget-segment customer"""
    agent = CustomerProfilingAgent(budget_customer_repo)

    input_data = CustomerProfilingInput(customer_id="1")
    output = await agent.run(input_data, agent_context)

    profile = output.profile

    # Assertions
    assert profile.customer_id == "1"
    assert profile.customer_name == "Budget Bob"
    assert profile.total_purchases == 3
    assert profile.total_spent == pytest.approx(50 + 60 + 70, abs=1)
    assert profile.avg_purchase_price < 200  # Budget range
    assert profile.price_segment == "budget"
    assert profile.purchase_frequency == "low"  # 3 purchases < MEDIUM_FREQUENCY_THRESHOLD
    assert "Electronics" in profile.favorite_categories
    assert len(profile.recent_purchases) > 0


@pytest.mark.asyncio
async def test_premium_customer_profiling(premium_customer_repo, agent_context):
    """Test profiling a premium-segment customer"""
    agent = CustomerProfilingAgent(premium_customer_repo)

    input_data = CustomerProfilingInput(customer_id="2")
    output = await agent.run(input_data, agent_context)

    profile = output.profile

    # Assertions
    assert profile.customer_id == "2"
    assert profile.customer_name == "Premium Petra"
    assert profile.total_purchases == 12
    assert profile.avg_purchase_price > 600  # Premium range
    assert profile.price_segment == "premium"
    assert profile.purchase_frequency == "high"  # 12 > HIGH_FREQUENCY_THRESHOLD
    assert "Electronics" in profile.favorite_categories
    assert profile.confidence == 1.0  # 12 purchases > 10, so capped at 1.0


@pytest.mark.asyncio
async def test_customer_not_found(budget_customer_repo, agent_context):
    """Test error handling when customer doesn't exist"""
    agent = CustomerProfilingAgent(budget_customer_repo)

    input_data = CustomerProfilingInput(customer_id="nonexistent")

    with pytest.raises(Exception):  # Should raise NotFoundException
        await agent.run(input_data, agent_context)


@pytest.mark.asyncio
async def test_favorite_categories_ordering():
    """Test that favorite categories are ordered by purchase count"""
    customers = [{"customer_id": "3", "customer_name": "Test User"}]
    purchases = {
        "3": [
            # 5 Electronics
            *[
                {
                    "transaction_id": f"t{i}",
                    "product_id": f"e{i}",
                    "product_name": "Electronics",
                    "product_category": "Electronics",
                    "price": 100.0,
                }
                for i in range(5)
            ],
            # 3 Books
            *[
                {
                    "transaction_id": f"t{i+5}",
                    "product_id": f"b{i}",
                    "product_name": "Book",
                    "product_category": "Books",
                    "price": 20.0,
                }
                for i in range(3)
            ],
            # 2 Clothing
            *[
                {
                    "transaction_id": f"t{i+8}",
                    "product_id": f"c{i}",
                    "product_name": "Shirt",
                    "product_category": "Clothing",
                    "price": 30.0,
                }
                for i in range(2)
            ],
        ]
    }
    repo = FakeCustomerRepository(customers, purchases)
    agent = CustomerProfilingAgent(repo)
    context = AgentContext(request_id="test", user_id="3")

    input_data = CustomerProfilingInput(customer_id="3")
    output = await agent.run(input_data, context)

    profile = output.profile

    # Favorite categories should be ordered: Electronics (5), Books (3), Clothing (2)
    assert profile.favorite_categories == ["Electronics", "Books", "Clothing"]


@pytest.mark.asyncio
async def test_mid_range_segment():
    """Test mid-range price segment classification"""
    customers = [{"customer_id": "4", "customer_name": "Mid User"}]
    purchases = {
        "4": [
            {
                "transaction_id": f"t{i}",
                "product_id": f"p{i}",
                "product_name": f"Product {i}",
                "product_category": "Electronics",
                "price": 400.0,  # Between 200 and 600
            }
            for i in range(5)
        ]
    }
    repo = FakeCustomerRepository(customers, purchases)
    agent = CustomerProfilingAgent(repo)
    context = AgentContext(request_id="test", user_id="4")

    input_data = CustomerProfilingInput(customer_id="4")
    output = await agent.run(input_data, context)

    profile = output.profile

    assert profile.price_segment == "mid-range"
    assert 200 <= profile.avg_purchase_price < 600


@pytest.mark.asyncio
async def test_medium_frequency_classification():
    """Test medium purchase frequency classification"""
    customers = [{"customer_id": "5", "customer_name": "Medium User"}]
    # 5 purchases: >= MEDIUM_FREQUENCY_THRESHOLD (3) but < HIGH_FREQUENCY_THRESHOLD (10)
    purchases = {
        "5": [
            {
                "transaction_id": f"t{i}",
                "product_id": f"p{i}",
                "product_name": f"Product {i}",
                "product_category": "Books",
                "price": 25.0,
            }
            for i in range(5)
        ]
    }
    repo = FakeCustomerRepository(customers, purchases)
    agent = CustomerProfilingAgent(repo)
    context = AgentContext(request_id="test", user_id="5")

    input_data = CustomerProfilingInput(customer_id="5")
    output = await agent.run(input_data, context)

    profile = output.profile

    assert profile.purchase_frequency == "medium"
    assert profile.total_purchases == 5
