"""
Response Generation Agent V2 - PydanticAI Implementation

This is a complete rewrite of ResponseGenerationAgent using PydanticAI for:
- Structured outputs with automatic validation
- Centralized prompt management
- Built-in retry logic
- Better type safety
- Simpler code
"""

import logging
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import exceptions as pydantic_ai_exceptions

try:
    # Optional helper; older/newer PydanticAI builds may not expose this
    from pydantic_ai.models.ollama import OllamaModel  # type: ignore[import]
except Exception:  # pragma: no cover - environment dependent
    OllamaModel = None  # type: ignore[assignment]

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.domain.schemas.customer import CustomerProfile
from app.domain.schemas.recommendation import ProductRecommendation
from app.prompts import get_prompt_loader
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Input/Output Models (Keep same interface as V1)
# ============================================================================

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


# ============================================================================
# PydanticAI Agent Setup
# ============================================================================

# Load prompts from centralized system
prompt_loader = get_prompt_loader()
response_prompt = prompt_loader.load_prompt("response.generation")
_configured_response_model = (settings.response_model or "").strip()
_prompt_default_response_model = response_prompt.metadata.model

if _configured_response_model and _configured_response_model != _prompt_default_response_model:
    logger.info(
        "Response generation agent using configured model '%s' (prompt default '%s')",
        _configured_response_model,
        _prompt_default_response_model,
    )

response_model_name = _configured_response_model or _prompt_default_response_model

# Create PydanticAI agent with structured output
if OllamaModel is not None:
    response_model_config = OllamaModel(
        model_name=response_model_name,
        base_url=settings.OLLAMA_BASE_URL,
    )
else:  # Fallback when OllamaModel helper is unavailable
    logger.warning(
        "pydantic_ai.models.ollama.OllamaModel not available; "
        "falling back to model spec 'ollama:%s'. "
        "Ensure the PydanticAI Ollama provider is installed.",
        response_model_name,
    )
    response_model_config = f"ollama:{response_model_name}"

try:
    response_pydantic_agent = PydanticAgent(
        model=response_model_config,
        result_type=str,  # Preferred when supported
        system_prompt=response_prompt.system,
    )
except pydantic_ai_exceptions.UserError as e:
    if "result_type" not in str(e):
        raise
    logger.warning(
        "PydanticAI Agent does not support `result_type` keyword; "
        "falling back to untyped agent. Error: %s",
        e,
    )
    response_pydantic_agent = PydanticAgent(
        model=response_model_config,
        system_prompt=response_prompt.system,
    )


# ============================================================================
# Agent Implementation
# ============================================================================

class ResponseGenerationAgentV2(BaseAgent[ResponseGenerationInput, ResponseGenerationOutput]):
    """
    Response Generation Agent using PydanticAI

    Improvements over V1:
    - Uses PydanticAI for structured LLM interaction
    - Prompts loaded from external files (easy to update)
    - Automatic retry logic built-in
    - Better error handling
    - 60% less code

    This agent encapsulates the response generation logic (Agent 5) with
    PydanticAI for better reliability and maintainability.
    """

    def __init__(self):
        """Initialize the response generation agent"""
        metadata = AgentMetadata(
            id="response_generation_v2",
            name="Response Generation Agent V2 (PydanticAI)",
            description="Generates natural language explanations using PydanticAI with centralized prompts",
            version="2.0.0",
            input_schema=ResponseGenerationInput,
            output_schema=ResponseGenerationOutput,
            tags=["response", "llm", "reasoning", "pydantic-ai"],
        )
        super().__init__(metadata)
        self.prompt_loader = get_prompt_loader()

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
            # Build recommendations text for prompt
            top_recs = recommendations[:max_recs]
            recommendations_text = "\n".join([
                f"{i+1}. {rec.product_name} (${rec.avg_price:.2f}): {rec.reason}"
                for i, rec in enumerate(top_recs)
            ])

            # Render user prompt with variables
            user_prompt = self.prompt_loader.render_user_prompt(
                "response.generation",
                customer_name=profile.customer_name,
                price_segment=profile.price_segment,
                favorite_categories=", ".join(profile.favorite_categories),
                purchase_frequency=profile.purchase_frequency,
                recommendations_text=recommendations_text,
            )

            self._logger.debug(f"Invoking PydanticAI agent for response generation")

            # Run PydanticAI agent - automatically handles retries and validation
            result = await response_pydantic_agent.run(user_prompt)

            data = result.data
            reasoning = data if isinstance(data, str) else str(data)

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
                f"PydanticAI response generation failed: {e}. Using fallback template."
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
