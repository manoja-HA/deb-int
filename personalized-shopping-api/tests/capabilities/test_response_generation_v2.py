"""
Tests for ResponseGenerationAgentV2 (PydanticAI version)

Note: These tests require Ollama to be running with the configured model.
For unit testing without LLM, mock the PydanticAgent.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.capabilities.base import AgentContext
from app.capabilities.agents.response_generation_v2 import (
    ResponseGenerationAgentV2,
    ResponseGenerationInput,
)
from app.domain.schemas.customer import CustomerProfile
from app.domain.schemas.recommendation import ProductRecommendation


@pytest.fixture
def sample_customer_profile():
    """Sample customer profile for testing"""
    return CustomerProfile(
        customer_id="123",
        customer_name="Test Customer",
        total_purchases=10,
        total_spent=5000.0,
        avg_purchase_price=500.0,
        favorite_categories=["Electronics", "Books"],
        favorite_brands=[],
        price_segment="mid-range",
        purchase_frequency="high",
        recent_purchases=[],
        confidence=1.0,
    )


@pytest.fixture
def sample_recommendations():
    """Sample recommendations for testing"""
    return [
        ProductRecommendation(
            product_id="1",
            product_name="Laptop",
            product_category="Electronics",
            avg_price=800.0,
            recommendation_score=0.95,
            reason="Highly popular with similar customers",
            similar_customer_count=10,
            avg_sentiment=0.9,
            source="collaborative",
        ),
        ProductRecommendation(
            product_id="2",
            product_name="Python Book",
            product_category="Books",
            avg_price=50.0,
            recommendation_score=0.85,
            reason="Matches your preference for Books",
            similar_customer_count=8,
            avg_sentiment=0.85,
            source="category_affinity",
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
async def test_agent_initialization():
    """Test that agent initializes correctly"""
    agent = ResponseGenerationAgentV2()

    assert agent.metadata.id == "response_generation_v2"
    assert agent.metadata.version == "2.0.0"
    assert "pydantic-ai" in agent.metadata.tags


@pytest.mark.asyncio
async def test_empty_recommendations_returns_fallback(
    sample_customer_profile,
    agent_context
):
    """Test that empty recommendations return fallback message"""
    agent = ResponseGenerationAgentV2()

    input_data = ResponseGenerationInput(
        query="What should I buy?",
        customer_profile=sample_customer_profile,
        recommendations=[],  # Empty!
    )

    output = await agent.run(input_data, agent_context)

    assert output.reasoning is not None
    assert "no suitable recommendations" in output.reasoning.lower()
    assert output.llm_used is False


@pytest.mark.asyncio
@patch('app.capabilities.agents.response_generation_v2.response_pydantic_agent')
async def test_successful_llm_response(
    mock_agent,
    sample_customer_profile,
    sample_recommendations,
    agent_context
):
    """Test successful LLM response generation (mocked)"""
    # Mock the PydanticAI agent response
    mock_result = Mock()
    mock_result.data = "These recommendations match your interests in Electronics and Books perfectly!"
    mock_agent.run = AsyncMock(return_value=mock_result)

    agent = ResponseGenerationAgentV2()

    input_data = ResponseGenerationInput(
        query="What should I buy?",
        customer_profile=sample_customer_profile,
        recommendations=sample_recommendations,
    )

    output = await agent.run(input_data, agent_context)

    # Verify output
    assert output.reasoning is not None
    assert len(output.reasoning) > 0
    assert output.llm_used is True

    # Verify mock was called
    mock_agent.run.assert_called_once()


@pytest.mark.asyncio
@patch('app.capabilities.agents.response_generation_v2.response_pydantic_agent')
async def test_llm_failure_returns_fallback(
    mock_agent,
    sample_customer_profile,
    sample_recommendations,
    agent_context
):
    """Test that LLM failure triggers fallback"""
    # Mock LLM failure
    mock_agent.run = AsyncMock(side_effect=Exception("LLM error"))

    agent = ResponseGenerationAgentV2()

    input_data = ResponseGenerationInput(
        query="What should I buy?",
        customer_profile=sample_customer_profile,
        recommendations=sample_recommendations,
    )

    output = await agent.run(input_data, agent_context)

    # Should return fallback
    assert output.reasoning is not None
    assert sample_customer_profile.customer_name in output.reasoning
    assert output.llm_used is False


@pytest.mark.asyncio
async def test_max_recommendations_limit(
    sample_customer_profile,
    agent_context
):
    """Test that only max_recommendations_to_explain are used"""
    # Create many recommendations
    recommendations = [
        ProductRecommendation(
            product_id=str(i),
            product_name=f"Product {i}",
            product_category="Electronics",
            avg_price=100.0 * i,
            recommendation_score=0.9,
            reason=f"Reason {i}",
            similar_customer_count=5,
            avg_sentiment=0.8,
            source="collaborative",
        )
        for i in range(10)
    ]

    with patch('app.capabilities.agents.response_generation_v2.response_pydantic_agent') as mock_agent:
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent.run = AsyncMock(return_value=mock_result)

        agent = ResponseGenerationAgentV2()

        input_data = ResponseGenerationInput(
            query="What should I buy?",
            customer_profile=sample_customer_profile,
            recommendations=recommendations,
            max_recommendations_to_explain=3,  # Limit to 3
        )

        await agent.run(input_data, agent_context)

        # Check that the user prompt only includes 3 recommendations
        call_args = mock_agent.run.call_args
        user_prompt = call_args[0][0]

        # Should only have "1.", "2.", "3." in the prompt
        assert "1. Product 0" in user_prompt
        assert "2. Product 1" in user_prompt
        assert "3. Product 2" in user_prompt
        assert "4. Product 3" not in user_prompt  # Should not include 4th


@pytest.mark.asyncio
async def test_observability_metadata(
    sample_customer_profile,
    sample_recommendations,
    agent_context
):
    """Test that execution metadata is recorded"""
    with patch('app.capabilities.agents.response_generation_v2.response_pydantic_agent') as mock_agent:
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent.run = AsyncMock(return_value=mock_result)

        agent = ResponseGenerationAgentV2()

        input_data = ResponseGenerationInput(
            query="What should I buy?",
            customer_profile=sample_customer_profile,
            recommendations=sample_recommendations,
        )

        await agent.run(input_data, agent_context)

        # Check execution metadata
        assert "agent_executions" in agent_context.metadata
        executions = agent_context.metadata["agent_executions"]
        assert len(executions) == 1

        execution = executions[0]
        assert execution["agent_id"] == "response_generation_v2"
        assert execution["success"] is True
        assert "execution_time_ms" in execution
        assert execution["execution_time_ms"] > 0
