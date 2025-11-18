"""
LLM factory for managing model instances with caching and tracing
"""

from typing import Dict, Literal, List, Optional, Any
from enum import Enum
import logging
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

from app.core.config import settings as config
from app.core.tracing import get_langfuse_callback

logger = logging.getLogger(__name__)

class LLMType(str, Enum):
    """LLM types for different tasks"""
    PROFILING = "profiling"
    SENTIMENT = "sentiment"
    RECOMMENDATION = "recommendation"
    RESPONSE = "response"
    INTENT = "intent"

# Global cache for LLM instances
_llm_cache: Dict[str, ChatOllama] = {}

def get_llm(
    llm_type: LLMType,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> ChatOllama:
    """
    Get or create LLM instance with caching and optional tracing

    Args:
        llm_type: Type of LLM to retrieve
        callbacks: Additional callbacks to include
        session_id: Session ID for tracing
        user_id: User ID for tracing
        metadata: Additional metadata for tracing
        tags: Tags for categorization

    Returns:
        ChatOllama instance configured for the task with tracing enabled

    Example:
        llm = get_llm(
            LLMType.RESPONSE,
            session_id="req-123",
            user_id="Kenneth Martinez",
            metadata={"query_type": "informational"},
            tags=["intent_classification"]
        )
        response = llm.invoke(messages)
    """
    # Map LLM type to model name
    model_mapping = {
        LLMType.PROFILING: config.profiling_model,
        LLMType.SENTIMENT: config.sentiment_model,
        LLMType.RECOMMENDATION: config.recommendation_model,
        LLMType.RESPONSE: config.response_model,
        LLMType.INTENT: getattr(config, 'intent_model', config.response_model),  # Fallback to response_model
    }

    model_name = model_mapping[llm_type]

    # Get cached LLM instance
    llm = _get_cached_llm(model_name)

    # Prepare callbacks
    all_callbacks = list(callbacks) if callbacks else []

    # Add LangFuse callback if tracing is enabled
    langfuse_callback = get_langfuse_callback(
        session_id=session_id,
        user_id=user_id,
        metadata={
            "llm_type": llm_type.value,
            "model": model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **(metadata or {})
        },
        tags=[llm_type.value, "ollama", model_name] + (tags or [])
    )

    if langfuse_callback:
        all_callbacks.append(langfuse_callback)
        logger.debug(f"Added LangFuse callback for {model_name}")

    # Return LLM with callbacks if any
    if all_callbacks:
        return llm.bind(callbacks=all_callbacks)

    return llm


@lru_cache(maxsize=4)
def _get_cached_llm(model_name: str) -> ChatOllama:
    """
    Internal function to get cached LLM instance

    Args:
        model_name: Name of the model

    Returns:
        Cached ChatOllama instance
    """
    logger.info(f"Initializing LLM: {model_name}")

    try:
        llm = ChatOllama(
            base_url=config.ollama_base_url,
            model=model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            timeout=config.request_timeout_seconds,
        )

        logger.info(f"Successfully initialized {model_name}")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM {model_name}: {e}")
        raise

def clear_llm_cache() -> None:
    """Clear the LLM cache"""
    global _llm_cache
    _llm_cache.clear()
    logger.info("Cleared LLM cache")

def get_model_info(llm_type: LLMType) -> Dict:
    """
    Get information about a model

    Args:
        llm_type: Type of LLM

    Returns:
        Dictionary with model information
    """
    model_mapping = {
        LLMType.PROFILING: config.profiling_model,
        LLMType.SENTIMENT: config.sentiment_model,
        LLMType.RECOMMENDATION: config.recommendation_model,
        LLMType.RESPONSE: config.response_model,
        LLMType.INTENT: getattr(config, 'intent_model', config.response_model),  # Fallback to response_model
    }

    return {
        'model_name': model_mapping[llm_type],
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
        'base_url': config.ollama_base_url,
    }
