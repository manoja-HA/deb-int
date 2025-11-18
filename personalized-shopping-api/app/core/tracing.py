"""
LangFuse tracing integration
Provides observability for LLM calls and agent workflows
"""

from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Conditional imports for LangFuse
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("LangFuse not installed. Install with: pip install langfuse")
    LANGFUSE_AVAILABLE = False
    Langfuse = None

# Try to import LangChain callback (optional)
try:
    from langfuse.callback import CallbackHandler
    LANGCHAIN_CALLBACK_AVAILABLE = True
except ImportError:
    logger.info("LangChain callback not available (langfuse-langchain not installed)")
    LANGCHAIN_CALLBACK_AVAILABLE = False
    CallbackHandler = None


class LangFuseTracer:
    """Singleton wrapper for LangFuse client"""

    _instance: Optional[Any] = None
    _enabled: bool = False
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        """Initialize LangFuse client"""
        if cls._initialized:
            return

        cls._initialized = True

        if not LANGFUSE_AVAILABLE:
            logger.info("LangFuse library not available")
            return

        if not settings.LANGFUSE_ENABLED:
            logger.info("LangFuse tracing is disabled (set LANGFUSE_ENABLED=true to enable)")
            return

        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            logger.warning(
                "LangFuse credentials not configured. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )
            return

        try:
            cls._instance = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                release=settings.LANGFUSE_RELEASE,
                environment=settings.LANGFUSE_ENVIRONMENT,
                debug=settings.LANGFUSE_DEBUG,
            )
            cls._enabled = True
            logger.info(
                f"LangFuse initialized successfully: {settings.LANGFUSE_HOST} "
                f"(environment: {settings.LANGFUSE_ENVIRONMENT})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}", exc_info=True)
            cls._enabled = False

    @classmethod
    def get_client(cls) -> Optional[Any]:
        """Get LangFuse client instance"""
        if not cls._initialized:
            cls.initialize()

        if not cls._enabled or not cls._instance:
            return None

        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracing is enabled"""
        if not cls._initialized:
            cls.initialize()

        return cls._enabled and LANGFUSE_AVAILABLE

    @classmethod
    def flush(cls):
        """Flush pending traces to LangFuse"""
        if cls._instance and hasattr(cls._instance, 'flush'):
            try:
                cls._instance.flush()
                logger.debug("Flushed LangFuse traces")
            except Exception as e:
                logger.error(f"Failed to flush LangFuse traces: {e}")


def get_langfuse_callback(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """
    Get LangChain callback handler for LangFuse

    Args:
        session_id: Session/trace identifier for grouping related calls
        user_id: User identifier (customer_id or customer_name)
        metadata: Additional metadata to attach to traces
        tags: Tags for categorization and filtering

    Returns:
        CallbackHandler if enabled, None otherwise

    Example:
        callback = get_langfuse_callback(
            session_id="req-123",
            user_id="Kenneth Martinez",
            metadata={"query_type": "informational"},
            tags=["intent_classification", "customer_887"]
        )
        response = llm.invoke(messages, config={"callbacks": [callback]})
    """
    if not LangFuseTracer.is_enabled() or not LANGFUSE_AVAILABLE:
        return None

    if not LANGCHAIN_CALLBACK_AVAILABLE:
        logger.warning("LangChain callback not available. Install langfuse-langchain for automatic LangChain tracing.")
        return None

    client = LangFuseTracer.get_client()
    if not client:
        return None

    try:
        return CallbackHandler(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
            langfuse=client,
        )
    except Exception as e:
        logger.error(f"Failed to create LangFuse callback: {e}")
        return None


@contextmanager
def trace_span(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    input_data: Optional[Any] = None,
    **kwargs
):
    """
    Context manager for manual span tracing

    Creates a trace span for non-LLM operations that you want to monitor.
    Automatically captures timing and handles errors.

    Args:
        name: Name of the span (e.g., "customer_profiling", "sentiment_analysis")
        user_id: User identifier
        session_id: Session identifier
        metadata: Additional metadata
        tags: Tags for categorization
        input_data: Input data to log
        **kwargs: Additional arguments passed to span creation

    Yields:
        Span object if enabled, None otherwise

    Example:
        with trace_span(
            "customer_profiling",
            user_id="887",
            metadata={"purchases": 5},
            tags=["agent", "profiling"]
        ) as span:
            profile = await get_customer_profile(customer_id)
            if span:
                span.update(output=profile)
    """
    client = LangFuseTracer.get_client()

    if not client:
        # If tracing is disabled, just execute the block
        yield None
        return

    # Prepare metadata with tags and user/session info
    combined_metadata = metadata or {}
    if tags:
        combined_metadata["tags"] = tags
    if user_id:
        combined_metadata["user_id"] = user_id
    if session_id:
        combined_metadata["session_id"] = session_id

    try:
        # Use start_as_current_span which is the Langfuse v3 API
        with client.start_as_current_span(
            name=name,
            input=input_data,
            metadata=combined_metadata,
            **kwargs
        ) as span:
            yield span

    except Exception as e:
        # Error is automatically logged by Langfuse
        logger.error(f"Error in trace span '{name}': {e}", exc_info=True)
        raise


def log_event(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    level: str = "DEFAULT",
    input_data: Optional[Any] = None,
    output_data: Optional[Any] = None,
):
    """
    Log a custom event to LangFuse

    Use this for logging important events that aren't LLM calls or spans,
    such as user actions, system events, or business logic milestones.

    Args:
        name: Event name
        metadata: Event metadata
        level: Event level (DEFAULT, DEBUG, WARNING, ERROR)
        input_data: Input data for the event
        output_data: Output data for the event

    Example:
        log_event(
            "product_filter_applied",
            metadata={
                "sentiment_threshold": 0.6,
                "products_filtered": 15,
                "products_remaining": 42
            }
        )
    """
    client = LangFuseTracer.get_client()
    if not client:
        return

    try:
        client.create_event(
            name=name,
            metadata=metadata or {},
            level=level,
            input=input_data,
            output=output_data,
        )
    except Exception as e:
        logger.error(f"Failed to log event to LangFuse: {e}")


def log_score(
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
    observation_id: Optional[str] = None,
):
    """
    Log a score/evaluation for a trace

    Use this to track quality metrics, user feedback, or automated evaluations.

    Args:
        trace_id: ID of the trace to score
        name: Score name (e.g., "user_rating", "relevance", "accuracy")
        value: Score value (typically 0-1 or 1-5)
        comment: Optional comment/explanation
        data_type: Type of score (NUMERIC, CATEGORICAL, BOOLEAN)
        observation_id: Optional ID of specific observation to score

    Example:
        # User feedback
        log_score(
            trace_id=request_id,
            name="user_rating",
            value=4.5,
            comment="Great recommendations!"
        )

        # Automated evaluation
        log_score(
            trace_id=request_id,
            name="intent_accuracy",
            value=0.95,
            comment="Intent correctly classified"
        )
    """
    client = LangFuseTracer.get_client()
    if not client:
        return

    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
            observation_id=observation_id,
        )
        logger.debug(f"Logged score to LangFuse: {name}={value} for trace {trace_id}")
    except Exception as e:
        logger.error(f"Failed to log score to LangFuse: {e}")


def create_generation(
    name: str,
    input_data: Any,
    output_data: Optional[Any] = None,
    model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    usage_details: Optional[Dict[str, int]] = None,
    **kwargs
):
    """
    Manually create a generation (LLM call) event as a context manager

    Use this when you need to manually log LLM calls that aren't automatically
    traced by the callback handler (e.g., custom API calls, external services).

    Args:
        name: Generation name
        input_data: Input prompt/messages
        output_data: Generated output
        model: Model identifier
        metadata: Additional metadata
        model_parameters: Model parameters (e.g., temperature, max_tokens)
        usage_details: Token usage (e.g., prompt_tokens, completion_tokens)
        **kwargs: Additional arguments

    Returns:
        Context manager yielding a LangfuseGeneration

    Example:
        with create_generation(
            name="custom_llm_call",
            input_data={"prompt": "What is AI?"},
            model="custom-model-v1",
            metadata={"temperature": 0.7}
        ) as generation:
            response = call_custom_llm()
            generation.update(
                output=response,
                usage_details={"prompt_tokens": 10, "completion_tokens": 150}
            )
    """
    client = LangFuseTracer.get_client()
    if not client:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    try:
        return client.start_as_current_generation(
            name=name,
            input=input_data,
            output=output_data,
            model=model,
            metadata=metadata or {},
            model_parameters=model_parameters,
            usage_details=usage_details,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create generation in LangFuse: {e}")
        from contextlib import nullcontext
        return nullcontext()


# Initialize LangFuse on module import
LangFuseTracer.initialize()
