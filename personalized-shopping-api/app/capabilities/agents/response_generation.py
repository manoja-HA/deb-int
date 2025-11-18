"""
Response Generation Agent

Generates natural language explanations for recommendations using an LLM.
This agent creates human-readable reasoning that explains why products were recommended.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import logging

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.domain.schemas.customer import CustomerProfile
from app.domain.schemas.recommendation import ProductRecommendation
from app.models.llm_factory import get_llm, LLMType

logger = logging.getLogger(__name__)


class ResponseGenerationInput(BaseModel):
    """Input for response generation agent"""

    query: str = Field(description="Original customer query")
    customer_profile: CustomerProfile = Field(
        description="Customer profile context"
    )
    recommendations: List[ProductRecommendation] = Field(
        description="Product recommendations to explain"
    )
    max_recommendations_to_explain: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of recommendations to include in explanation"
    )


class ResponseGenerationOutput(BaseModel):
    """Output from response generation agent"""

    reasoning: str = Field(
        description="Natural language explanation of recommendations"
    )
    llm_used: bool = Field(
        default=False,
        description="Whether LLM was used (vs fallback template)"
    )


class ResponseGenerationAgent(BaseAgent[ResponseGenerationInput, ResponseGenerationOutput]):
    """
    Agent that generates natural language reasoning for recommendations

    Responsibilities:
    - Build context from customer profile and recommendations
    - Construct LLM prompt with customer context and top recommendations
    - Invoke LLM to generate human-readable explanation
    - Handle LLM failures with fallback template
    - Support tracing for LLM calls

    This agent encapsulates the response generation logic (Agent 5) that was
    previously embedded in RecommendationService._generate_reasoning().
    """

    def __init__(self):
        """Initialize the response generation agent"""
        metadata = AgentMetadata(
            id="response_generation",
            name="Response Generation Agent",
            description="Generates natural language explanations for recommendations using LLM",
            version="1.0.0",
            input_schema=ResponseGenerationInput,
            output_schema=ResponseGenerationOutput,
            tags=["response", "llm", "reasoning", "explanation"],
        )
        super().__init__(metadata)

    async def _execute(
        self,
        input_data: ResponseGenerationInput,
        context: AgentContext,
    ) -> ResponseGenerationOutput:
        """
        Generate natural language reasoning for recommendations

        Args:
            input_data: Contains query, customer profile, and recommendations
            context: Execution context with session_id for tracing

        Returns:
            Natural language explanation
        """
        query = input_data.query
        profile = input_data.customer_profile
        recommendations = input_data.recommendations
        max_recs = input_data.max_recommendations_to_explain

        # If no recommendations, return simple message
        if not recommendations:
            return ResponseGenerationOutput(
                reasoning=f"Unfortunately, no suitable recommendations were found for {profile.customer_name} at this time.",
                llm_used=False,
            )

        try:
            # Build customer context
            customer_context = (
                f"Customer: {profile.customer_name}, "
                f"{profile.purchase_frequency} buyer, "
                f"{profile.price_segment} segment, "
                f"Favorite categories: {', '.join(profile.favorite_categories)}"
            )

            # Build recommendation summary (top N)
            top_recs = recommendations[:max_recs]
            rec_text = "\n".join([
                f"{i+1}. {rec.product_name} (${rec.avg_price:.2f}): {rec.reason}"
                for i, rec in enumerate(top_recs)
            ])

            # Construct LLM prompt
            prompt = f"""Based on the customer's purchase history, provide a brief explanation for these recommendations.

{customer_context}

Recommendations:
{rec_text}

Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences."""

            # Get LLM with tracing
            llm = get_llm(
                LLMType.RESPONSE,
                session_id=context.session_id or context.request_id,
                user_id=profile.customer_name,
                metadata={
                    "task": "reasoning_generation",
                    "recommendations_count": len(recommendations),
                },
                tags=["reasoning", "response_generation"]
            )

            # Invoke LLM
            self._logger.debug(f"Invoking LLM for response generation")
            response = llm.invoke([HumanMessage(content=prompt)])

            reasoning = response.content.strip()

            self._logger.info(
                f"Generated reasoning ({len(reasoning)} chars) for {len(recommendations)} recommendations"
            )

            return ResponseGenerationOutput(
                reasoning=reasoning,
                llm_used=True,
            )

        except Exception as e:
            # Fallback to template-based response
            self._logger.warning(
                f"LLM response generation failed: {e}. Using fallback template."
            )

            fallback_reasoning = (
                f"Based on {profile.customer_name}'s purchase history as a "
                f"{profile.purchase_frequency} {profile.price_segment} buyer, "
                f"here are {len(recommendations)} recommendations that match their "
                f"preferences in {', '.join(profile.favorite_categories)}."
            )

            return ResponseGenerationOutput(
                reasoning=fallback_reasoning,
                llm_used=False,
            )
