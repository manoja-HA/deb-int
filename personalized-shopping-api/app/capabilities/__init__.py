"""
Capabilities package - Reusable agent components

This package contains the base agent architecture and concrete agent implementations
that follow a uniform interface for composability and testability.
"""

from app.capabilities.base import (
    AgentContext,
    BaseAgent,
    AgentMetadata,
    AgentRegistry,
)

__all__ = [
    "AgentContext",
    "BaseAgent",
    "AgentMetadata",
    "AgentRegistry",
]
