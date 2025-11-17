"""
LLM factory for managing model instances with caching
"""

from typing import Dict, Literal
from enum import Enum
import logging
from functools import lru_cache

from langchain_ollama import ChatOllama
from app.core.config import settings as config

logger = logging.getLogger(__name__)

class LLMType(str, Enum):
    """LLM types for different tasks"""
    PROFILING = "profiling"
    SENTIMENT = "sentiment"
    RECOMMENDATION = "recommendation"
    RESPONSE = "response"

# Global cache for LLM instances
_llm_cache: Dict[str, ChatOllama] = {}

@lru_cache(maxsize=4)
def get_llm(llm_type: LLMType) -> ChatOllama:
    """
    Get or create LLM instance with caching

    Args:
        llm_type: Type of LLM to retrieve

    Returns:
        ChatOllama instance configured for the task
    """
    # Map LLM type to model name
    model_mapping = {
        LLMType.PROFILING: config.profiling_model,
        LLMType.SENTIMENT: config.sentiment_model,
        LLMType.RECOMMENDATION: config.recommendation_model,
        LLMType.RESPONSE: config.response_model,
    }

    model_name = model_mapping[llm_type]

    # Check cache
    if model_name in _llm_cache:
        logger.debug(f"Using cached LLM: {model_name}")
        return _llm_cache[model_name]

    # Create new instance
    logger.info(f"Initializing LLM: {model_name}")

    try:
        llm = ChatOllama(
            base_url=config.ollama_base_url,
            model=model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            timeout=config.request_timeout_seconds,
        )

        # Cache instance
        _llm_cache[model_name] = llm

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
    }

    return {
        'model_name': model_mapping[llm_type],
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
        'base_url': config.ollama_base_url,
    }
