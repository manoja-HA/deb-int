"""LLM infrastructure module"""

from enum import Enum
from typing import Optional
import logging

from langchain_ollama import ChatOllama
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMType(str, Enum):
    """LLM model types for different tasks"""
    PROFILING = "profiling"
    SENTIMENT = "sentiment"
    RECOMMENDATION = "recommendation"
    RESPONSE = "response"


class LLMFactory:
    """Factory for creating LLM instances"""

    _instances: dict = {}

    @classmethod
    def get_llm(cls, llm_type: LLMType, temperature: float = 0.1) -> ChatOllama:
        """Get or create LLM instance

        Args:
            llm_type: Type of LLM to create
            temperature: Temperature for generation

        Returns:
            ChatOllama instance
        """
        cache_key = f"{llm_type}_{temperature}"

        if cache_key not in cls._instances:
            model_name = cls._get_model_name(llm_type)
            logger.info(f"Creating LLM instance: {model_name} (type={llm_type}, temp={temperature})")

            cls._instances[cache_key] = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_name,
                temperature=temperature,
            )

        return cls._instances[cache_key]

    @staticmethod
    def _get_model_name(llm_type: LLMType) -> str:
        """Get model name for LLM type"""
        model_map = {
            LLMType.PROFILING: settings.PROFILING_MODEL,
            LLMType.SENTIMENT: settings.SENTIMENT_MODEL,
            LLMType.RECOMMENDATION: settings.RECOMMENDATION_MODEL,
            LLMType.RESPONSE: settings.RESPONSE_MODEL,
        }
        return model_map.get(llm_type, settings.RESPONSE_MODEL)


def get_llm(llm_type: LLMType, temperature: float = 0.1) -> ChatOllama:
    """Convenience function to get LLM instance

    Args:
        llm_type: Type of LLM
        temperature: Temperature for generation

    Returns:
        ChatOllama instance
    """
    return LLMFactory.get_llm(llm_type, temperature)
