"""
Agent 5: Response Generation Agent
Format recommendations as natural language response
"""

from typing import Dict
import logging
from langchain_core.messages import HumanMessage

from ..state import ShoppingAssistantState
from ..models.llm_factory import get_llm, LLMType
from ..config import config
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)

@track_agent_performance("response_generation")
def response_generation_agent(state: ShoppingAssistantState) -> Dict:
    """
    Agent 5: Generate natural language response with recommendations

    Steps:
    1. Format recommendations as structured data
    2. Create prompt with customer context
    3. Generate conversational response using LLM
    4. Include metadata and reasoning

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields
    """
    recommendations = state.get("final_recommendations", [])
    profile = state.get("customer_profile")
    query = state.get("query", "")

    if not recommendations:
        logger.warning("No recommendations to format")

        # Generate fallback response
        fallback_response = _generate_fallback_response(state)

        return {
            "final_response": fallback_response,
            "response_format": "conversational",
            "agent_execution_order": ["response_generation"]
        }

    logger.info(f"Generating response for {len(recommendations)} recommendations")

    try:
        # Build context for LLM
        customer_context = _build_customer_context(profile)
        recommendations_text = _format_recommendations(recommendations)

        # Create prompt
        prompt = f"""You are a helpful shopping assistant. Based on the customer's purchase history and preferences, provide personalized product recommendations.

Customer Context:
{customer_context}

Original Query: "{query}"

Recommendations:
{recommendations_text}

Generate a friendly, conversational response that:
1. Directly addresses the customer's query
2. Lists the top {len(recommendations)} recommendations with brief explanations
3. Explains WHY each product is recommended based on the customer's preferences
4. Includes relevant details like price and review quality
5. Maintains a helpful, professional tone

Format your response naturally, as if speaking to the customer."""

        # Generate response using LLM
        llm = get_llm(LLMType.RESPONSE)
        response = llm.invoke([HumanMessage(content=prompt)])

        final_response = response.content.strip()

        logger.info(f"Generated response ({len(final_response)} characters)")

        return {
            "final_response": final_response,
            "response_format": "conversational",
            "agent_execution_order": ["response_generation"]
        }

    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)

        # Use simple fallback formatting
        fallback_response = _generate_simple_response(recommendations, profile)

        return {
            "warnings": [f"LLM response generation failed: {str(e)}"],
            "final_response": fallback_response,
            "response_format": "structured",
            "fallback_used": True,
            "agent_execution_order": ["response_generation"]
        }

def _build_customer_context(profile: Dict) -> str:
    """Build customer context summary"""
    if not profile:
        return "New customer with limited history"

    context = f"""- Customer: {profile['customer_name']}
- Purchase Frequency: {profile['purchase_frequency']}
- Price Segment: {profile['price_segment']}
- Favorite Categories: {', '.join(profile['favorite_categories'])}
- Total Purchases: {profile['total_purchases']}
- Average Purchase: ${profile['avg_purchase_price']:.2f}"""

    return context

def _format_recommendations(recommendations: list) -> str:
    """Format recommendations for prompt"""
    lines = []

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec['product_name']}")
        lines.append(f"   Category: {rec['product_category']}")
        lines.append(f"   Price: ${rec['avg_price']:.2f}")
        lines.append(f"   Reason: {rec['reason']}")
        lines.append(f"   Sentiment Score: {rec['avg_sentiment']:.0%}")
        lines.append(f"   Popularity: {rec['similar_customer_count']} similar customers")
        lines.append("")

    return "\n".join(lines)

def _generate_fallback_response(state: Dict) -> str:
    """Generate fallback response when no recommendations"""
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])

    if errors:
        return f"I apologize, but I encountered an issue: {errors[0]}"

    if warnings:
        return f"I couldn't generate recommendations: {warnings[0]}"

    return "I couldn't find suitable recommendations at this time. Please try again with different criteria."

def _generate_simple_response(recommendations: list, profile: Dict) -> str:
    """Generate simple formatted response without LLM"""
    lines = [
        f"Based on {profile['customer_name'] if profile else 'your'} purchase history, "
        f"here are my recommendations:\n"
    ]

    for i, rec in enumerate(recommendations, 1):
        sentiment_desc = "excellent" if rec['avg_sentiment'] > 0.8 else "positive"

        lines.append(
            f"{i}. **{rec['product_name']}** (${rec['avg_price']:.2f})"
        )
        lines.append(
            f"   {rec['reason']}. "
            f"This product has {sentiment_desc} reviews from customers."
        )
        lines.append("")

    lines.append(
        f"\nThese recommendations are based on purchase patterns from "
        f"{len(state.get('similar_customers', []))} customers with similar preferences."
    )

    return "\n".join(lines)
