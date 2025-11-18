"""
Base agent architecture for the personalized shopping API

This module defines the foundational classes for building reusable, composable agents:
- AgentContext: Request-scoped context passed to all agents
- AgentMetadata: Agent identification and description
- BaseAgent: Generic base class for all agents with uniform interface
- AgentRegistry: Discovery mechanism for runtime agent introspection
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

# Generic types for agent input and output
InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


class AgentContext(BaseModel):
    """
    Execution context passed to all agents

    Contains request-scoped information like tracing IDs, user info,
    and metadata for observability and debugging.
    """

    request_id: str = Field(
        description="Unique identifier for the request/workflow execution"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracing (e.g., LangFuse session)"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User/customer identifier for this request"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Context creation timestamp"
    )

    class Config:
        frozen = False  # Allow modification for passing state between agents


class AgentMetadata(BaseModel):
    """
    Agent identification and description metadata

    Used for agent registry, discovery, and documentation.
    """

    id: str = Field(description="Unique agent identifier (snake_case)")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="What this agent does")
    version: str = Field(default="1.0.0", description="Agent version")
    input_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for input validation"
    )
    output_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for output validation"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g., 'recommendation', 'sentiment')"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Type[BaseModel] fields


class BaseAgent(Generic[InputModel, OutputModel], ABC):
    """
    Base class for all agents in the system

    Agents are single-purpose, reusable components that:
    - Accept strongly-typed Pydantic input
    - Return strongly-typed Pydantic output
    - Have clear metadata (id, name, description)
    - Support observability (timing, logging, tracing)
    - Are stateless (all state in input/context)

    Usage:
        class MyAgent(BaseAgent[MyInput, MyOutput]):
            def __init__(self, dependencies):
                metadata = AgentMetadata(
                    id="my_agent",
                    name="My Agent",
                    description="Does something useful"
                )
                super().__init__(metadata)
                self.dependencies = dependencies

            async def _execute(
                self,
                input_data: MyInput,
                context: AgentContext
            ) -> MyOutput:
                # Agent logic here
                return MyOutput(...)
    """

    def __init__(self, metadata: AgentMetadata):
        """
        Initialize the agent with metadata

        Args:
            metadata: Agent identification and schema information
        """
        self.metadata = metadata
        self._logger = logging.getLogger(f"agent.{metadata.id}")

    @abstractmethod
    async def _execute(
        self,
        input_data: InputModel,
        context: AgentContext,
    ) -> OutputModel:
        """
        Core agent logic (implemented by subclasses)

        Args:
            input_data: Validated input conforming to InputModel schema
            context: Request-scoped execution context

        Returns:
            Validated output conforming to OutputModel schema
        """
        pass

    async def run(
        self,
        input_data: InputModel,
        context: AgentContext,
    ) -> OutputModel:
        """
        Execute the agent with observability hooks

        This method wraps _execute() with:
        - Timing measurements
        - Logging
        - Error handling
        - Future: Tracing integration

        Args:
            input_data: Agent input (validated against input schema)
            context: Execution context

        Returns:
            Agent output (validated against output schema)

        Raises:
            Exception: Any error during agent execution (logged and re-raised)
        """
        start_time = time.time()
        agent_id = self.metadata.id

        self._logger.info(
            f"Agent '{agent_id}' starting",
            extra={
                "agent_id": agent_id,
                "request_id": context.request_id,
                "session_id": context.session_id,
            }
        )

        try:
            # Execute core logic
            output = await self._execute(input_data, context)

            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # milliseconds

            self._logger.info(
                f"Agent '{agent_id}' completed successfully",
                extra={
                    "agent_id": agent_id,
                    "request_id": context.request_id,
                    "execution_time_ms": execution_time,
                }
            )

            # Store execution metadata in context for downstream agents
            if "agent_executions" not in context.metadata:
                context.metadata["agent_executions"] = []

            context.metadata["agent_executions"].append({
                "agent_id": agent_id,
                "execution_time_ms": execution_time,
                "success": True,
            })

            return output

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self._logger.error(
                f"Agent '{agent_id}' failed",
                extra={
                    "agent_id": agent_id,
                    "request_id": context.request_id,
                    "execution_time_ms": execution_time,
                    "error": str(e),
                },
                exc_info=True
            )

            # Store failure metadata
            if "agent_executions" not in context.metadata:
                context.metadata["agent_executions"] = []

            context.metadata["agent_executions"].append({
                "agent_id": agent_id,
                "execution_time_ms": execution_time,
                "success": False,
                "error": str(e),
            })

            raise

    def get_input_schema(self) -> Optional[Type[BaseModel]]:
        """Get the Pydantic model for agent input"""
        return self.metadata.input_schema

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        """Get the Pydantic model for agent output"""
        return self.metadata.output_schema

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.metadata.id}')>"


class AgentRegistry:
    """
    Registry for agent discovery and introspection

    Allows runtime discovery of available agents, their schemas,
    and metadata. Useful for building UI/editors (n8n-style) and
    dynamic workflow composition.

    Usage:
        # Register an agent
        AgentRegistry.register(MyAgent, MyAgent.metadata)

        # Discover agents
        all_agents = AgentRegistry.list_agents()
        agent_metadata = AgentRegistry.get_metadata("my_agent")

        # Get agent class
        agent_class = AgentRegistry.get_agent("my_agent")
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        agent_class: Type[BaseAgent],
        metadata: AgentMetadata,
    ) -> None:
        """
        Register an agent for discovery

        Args:
            agent_class: The agent class
            metadata: Agent metadata (id, name, description, schemas)
        """
        agent_id = metadata.id

        if agent_id in cls._registry:
            logger.warning(f"Agent '{agent_id}' already registered, overwriting")

        cls._registry[agent_id] = {
            "class": agent_class,
            "metadata": metadata,
        }

        logger.debug(f"Registered agent: {agent_id}")

    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by ID"""
        entry = cls._registry.get(agent_id)
        return entry["class"] if entry else None

    @classmethod
    def get_metadata(cls, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID"""
        entry = cls._registry.get(agent_id)
        return entry["metadata"] if entry else None

    @classmethod
    def list_agents(cls) -> List[AgentMetadata]:
        """List all registered agents"""
        return [entry["metadata"] for entry in cls._registry.values()]

    @classmethod
    def find_by_tag(cls, tag: str) -> List[AgentMetadata]:
        """Find agents by tag"""
        return [
            entry["metadata"]
            for entry in cls._registry.values()
            if tag in entry["metadata"].tags
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)"""
        cls._registry.clear()
