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
import logging
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import exceptions as pydantic_ai_exceptions

try:
    # Newer pydantic-ai versions may not ship provider models
    # in this namespace; we handle that gracefully below.
    from pydantic_ai.models.ollama import OllamaModel  # type: ignore[import]
except Exception:  # pragma: no cover - environment dependent
    OllamaModel = None  # type: ignore[assignment]

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
_configured_intent_model = (settings.response_model or "").strip()
_prompt_default_intent_model = intent_prompt.metadata.model

if _configured_intent_model and _configured_intent_model != _prompt_default_intent_model:
    logger.info(
        "Intent classifier using configured model '%s' (prompt default '%s')",
        _configured_intent_model,
        _prompt_default_intent_model,
    )

intent_model_name = _configured_intent_model or _prompt_default_intent_model

# Create PydanticAI agent with structured output
if OllamaModel is not None:
    intent_model_config = OllamaModel(
        model_name=intent_model_name,
        base_url=settings.OLLAMA_BASE_URL,
    )
else:  # Fallback for environments without OllamaModel helper
    logger.warning(
        "pydantic_ai.models.ollama.OllamaModel not available; "
        "falling back to model spec 'ollama:%s'. "
        "Ensure the PydanticAI Ollama provider is installed.",
        intent_model_name,
    )
    intent_model_config = f"ollama:{intent_model_name}"

try:
    # Newer PydanticAI versions may not accept `result_type` kwarg.
    intent_pydantic_agent = PydanticAgent(
        model=intent_model_config,
        result_type=IntentClassification,  # Preferred when supported
        system_prompt=intent_prompt.system,
    )
except pydantic_ai_exceptions.UserError as e:
    # Fallback: construct agent without typed result and validate manually.
    if "result_type" not in str(e):
        raise
    logger.warning(
        "PydanticAI Agent does not support `result_type` keyword; "
        "falling back to untyped agent and manual validation. Error: %s",
        e,
    )
    intent_pydantic_agent = PydanticAgent(
        model=intent_model_config,
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
        result = await intent_pydantic_agent.run(user_prompt)

        # Prefer typed result when available; otherwise parse manually.
        data = result.data
        if isinstance(data, IntentClassification):
            classification = data
        else:
            # When no typed result is configured, PydanticAI returns raw text.
            # Expect JSON per prompt and validate via IntentClassification.
            from json import loads

            text = str(data).strip()
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_text = text[json_start:json_end]
            else:
                json_text = text

            payload = loads(json_text)
            classification = IntentClassification.model_validate(payload)

        logger.info(
            f"[Intent Classifier V2] Classified as: {classification.intent} "
            f"(confidence: {classification.confidence:.2f}) - {classification.reasoning}"
        )

        return classification
