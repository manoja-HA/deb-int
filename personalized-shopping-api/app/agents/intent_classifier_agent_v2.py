"""
Intent Classification Agent V2 - PydanticAI Implementation

Complete rewrite using PydanticAI for:
- Structured outputs (no manual JSON parsing!)
- Automatic validation
- Type-safe results
- Centralized prompt management
- 80% less code
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.ollama import OllamaModel
import logging

from app.prompts import get_prompt_loader
from app.core.config import settings
from app.agents.intent_classifier_agent import QueryIntent, InformationCategory

logger = logging.getLogger(__name__)


# ============================================================================
# Structured Output Models
# ============================================================================

class IntentClassification(BaseModel):
    """
    Structured intent classification result

    This replaces manual JSON parsing with automatic Pydantic validation
    Now uses the same enums as the original agent for compatibility
    """
    intent: QueryIntent = Field(
        description="Query intent type"
    )
    category: Optional[InformationCategory] = Field(
        default=None,
        description="Information category (for informational queries only)"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Classification confidence score"
    )
    reasoning: str = Field(
        min_length=10,
        description="Brief explanation of classification decision"
    )
    extracted_info: dict = Field(
        default_factory=dict,
        description="Any extracted entities or info from query"
    )

    @field_validator('intent', mode='before')
    @classmethod
    def validate_intent(cls, v):
        """Convert string to QueryIntent enum if needed"""
        if isinstance(v, str):
            return QueryIntent(v)
        return v

    @field_validator('category', mode='before')
    @classmethod
    def validate_category(cls, v):
        """Convert string to InformationCategory enum if needed"""
        if v is None:
            return None
        if isinstance(v, str):
            return InformationCategory(v)
        return v


# ============================================================================
# PydanticAI Agent Setup
# ============================================================================

# Load prompts from centralized system
prompt_loader = get_prompt_loader()
intent_prompt = prompt_loader.load_prompt("intent.classification")

# Create PydanticAI agent with structured output
intent_pydantic_agent = PydanticAgent(
    model=OllamaModel(
        model_name=intent_prompt.metadata.model,
        base_url=settings.OLLAMA_BASE_URL,
    ),
    result_type=IntentClassification,  # ✅ Automatic validation!
    system_prompt=intent_prompt.system,
)


# ============================================================================
# Agent Implementation
# ============================================================================

class IntentClassifierAgentV2:
    """
    Intent Classification Agent using PydanticAI

    Improvements over V1:
    - No manual JSON parsing (automatic with Pydantic)
    - Type-safe results
    - Prompts loaded from files
    - Built-in retry logic
    - 80% less code (50+ lines → 10 lines)

    Usage:
        agent = IntentClassifierAgentV2()
        result = await agent.classify("What should I buy?")
        # result is IntentClassification (fully typed!)
    """

    def __init__(self):
        """Initialize the intent classifier"""
        self.prompt_loader = get_prompt_loader()

    async def classify(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> IntentClassification:
        """
        Classify query intent with structured output

        Args:
            query: User query to classify
            session_id: Optional session ID for tracing
            user_id: Optional user ID for tracing

        Returns:
            Fully validated IntentClassification

        Raises:
            Exception: If classification fails after retries
        """
        logger.info(f"[Intent Classifier V2] Classifying query: '{query[:50]}...'")

        # Render user prompt
        user_prompt = self.prompt_loader.render_user_prompt(
            "intent.classification",
            query=query
        )

        # Run PydanticAI agent
        # Automatically:
        # - Invokes LLM
        # - Parses JSON response
        # - Validates against IntentClassification schema
        # - Retries on failure
        result = await intent_pydantic_agent.run(user_prompt)

        # result.data is IntentClassification (fully typed and validated!)
        classification = result.data

        logger.info(
            f"[Intent Classifier V2] Classified as: {classification.intent} "
            f"(confidence: {classification.confidence:.2f}) - {classification.reasoning}"
        )

        return classification
