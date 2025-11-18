# LangFuse Integration Guide

## Overview

This document outlines how to integrate **LangFuse** (an open-source LLM observability and analytics platform) into the Personalized Shopping Assistant API. LangFuse provides comprehensive tracing, monitoring, and analytics for LLM applications.

## Current State

### Observability Stack
Currently, the application uses:
- **LangSmith**: Basic tracing capability (configured but not actively used)
  - Config: `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
  - Status: Tracing disabled by default (`ENABLE_TRACING=false`)
- **Prometheus**: Metrics collection for API performance
- **Structured Logging**: JSON-formatted logs for debugging

### Why LangFuse?

**LangFuse** offers several advantages:
1. **Open Source**: Self-hostable, no vendor lock-in
2. **Comprehensive Tracing**: Full LLM call traces with prompts, completions, and metadata
3. **Cost Tracking**: Track token usage and costs per request
4. **User Analytics**: Understand user behavior and query patterns
5. **Prompt Management**: Version and manage prompts
6. **Evaluation**: Built-in tools for evaluating LLM outputs
7. **LangChain Integration**: Native support for LangChain/LangGraph

---

## Implementation Plan

### 1. Dependencies

Add LangFuse to `requirements.txt`:

```python
# Observability and Tracing
langfuse>=2.0.0
langfuse-langchain>=1.0.0  # For LangChain integration
```

### 2. Configuration

Update `.env`:

```bash
# LangFuse Configuration
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL

# Optional: LangFuse Project Settings
LANGFUSE_RELEASE=production
LANGFUSE_ENVIRONMENT=production
LANGFUSE_DEBUG=false
```

Update [app/core/config.py](personalized-shopping-api/app/core/config.py#L154-L162):

```python
# ===== OBSERVABILITY =====
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "json"
ENABLE_METRICS: bool = True
ENABLE_TRACING: bool = False

# LangSmith (Legacy - can be deprecated)
LANGSMITH_API_KEY: Optional[str] = None
LANGSMITH_PROJECT: str = "shopping-assistant-api"

# LangFuse (Recommended)
LANGFUSE_ENABLED: bool = False
LANGFUSE_PUBLIC_KEY: Optional[str] = None
LANGFUSE_SECRET_KEY: Optional[str] = None
LANGFUSE_HOST: str = "https://cloud.langfuse.com"
LANGFUSE_RELEASE: Optional[str] = None
LANGFUSE_ENVIRONMENT: str = "development"
LANGFUSE_DEBUG: bool = False
```

### 3. Core Integration

Create `app/core/tracing.py`:

```python
"""
LangFuse tracing integration
Provides observability for LLM calls and agent workflows
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

from app.core.config import settings

logger = logging.getLogger(__name__)


class LangFuseTracer:
    """Singleton wrapper for LangFuse client"""

    _instance: Optional[Langfuse] = None
    _enabled: bool = False

    @classmethod
    def initialize(cls):
        """Initialize LangFuse client"""
        if not settings.LANGFUSE_ENABLED:
            logger.info("LangFuse tracing is disabled")
            return

        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            logger.warning("LangFuse credentials not configured")
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
            logger.info(f"LangFuse initialized: {settings.LANGFUSE_HOST}")
        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}")

    @classmethod
    def get_client(cls) -> Optional[Langfuse]:
        """Get LangFuse client instance"""
        if not cls._enabled or not cls._instance:
            return None
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracing is enabled"""
        return cls._enabled

    @classmethod
    def flush(cls):
        """Flush pending traces"""
        if cls._instance:
            cls._instance.flush()


def get_langfuse_callback(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> Optional[BaseCallbackHandler]:
    """
    Get LangChain callback handler for LangFuse

    Args:
        session_id: Session/trace identifier
        user_id: User identifier (customer_id or customer_name)
        metadata: Additional metadata to attach
        tags: Tags for categorization

    Returns:
        CallbackHandler if enabled, None otherwise
    """
    if not LangFuseTracer.is_enabled():
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
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Context manager for manual span tracing

    Example:
        with trace_span("customer_profiling", user_id="887", metadata={"purchases": 5}):
            profile = await get_customer_profile(customer_id)
    """
    client = LangFuseTracer.get_client()
    if not client:
        yield None
        return

    trace = client.trace(
        name=name,
        user_id=user_id,
        metadata=metadata,
        **kwargs
    )

    try:
        yield trace
    finally:
        trace.update(status="completed")


# Initialize on module import
LangFuseTracer.initialize()
```

### 4. Update LLM Factory

Modify [app/models/llm_factory.py](personalized-shopping-api/app/models/llm_factory.py) to include LangFuse callbacks:

```python
"""
LLM factory for managing model instances with caching and tracing
"""

from typing import Dict, Literal, List, Optional
from enum import Enum
import logging
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

from app.core.config import settings as config
from app.core.tracing import get_langfuse_callback

logger = logging.getLogger(__name__)


def get_llm(
    llm_type: LLMType,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> ChatOllama:
    """
    Get or create LLM instance with caching and optional tracing

    Args:
        llm_type: Type of LLM to retrieve
        callbacks: Additional callbacks to include
        session_id: Session ID for tracing
        user_id: User ID for tracing
        metadata: Additional metadata for tracing

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

    # Get cached instance
    llm = _get_cached_llm(model_name)

    # Prepare callbacks
    all_callbacks = callbacks or []

    # Add LangFuse callback if tracing is enabled
    langfuse_callback = get_langfuse_callback(
        session_id=session_id,
        user_id=user_id,
        metadata={
            "llm_type": llm_type.value,
            "model": model_name,
            **(metadata or {})
        },
        tags=[llm_type.value, "ollama"]
    )

    if langfuse_callback:
        all_callbacks.append(langfuse_callback)

    # Return LLM with callbacks
    if all_callbacks:
        return llm.bind(callbacks=all_callbacks)

    return llm


@lru_cache(maxsize=4)
def _get_cached_llm(model_name: str) -> ChatOllama:
    """Internal function to get cached LLM instance"""
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
```

### 5. Update Intent Classifier Agent

Modify [app/agents/intent_classifier_agent.py](personalized-shopping-api/app/agents/intent_classifier_agent.py) to use tracing:

```python
def _classify_intent_node(self, state: AgentState) -> AgentState:
    """
    Node to classify the intent using LLM with tracing
    """
    query = state["query"]
    logger.info(f"[Intent Agent] Classifying query: '{query[:50]}...'")

    # Get LLM with tracing
    llm = self._get_llm()

    # Add LangFuse callback with metadata
    from app.core.tracing import get_langfuse_callback
    callback = get_langfuse_callback(
        session_id=state.get("session_id"),
        user_id=state.get("user_id"),
        metadata={
            "agent": "intent_classifier",
            "query": query,
            "node": "classify_intent"
        },
        tags=["intent_classification", "query_routing"]
    )

    callbacks = [callback] if callback else []

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]

        # Invoke with callbacks
        response = llm.invoke(messages, config={"callbacks": callbacks})

        # ... rest of the logic
    except Exception as e:
        logger.error(f"[Intent Agent] Classification failed: {e}")
        # ... fallback logic
```

### 6. Update Recommendation Service

Modify [app/services/recommendation_service.py](personalized-shopping-api/app/services/recommendation_service.py) to trace the entire workflow:

```python
from app.core.tracing import trace_span, LangFuseTracer
import uuid

async def get_personalized_recommendations(
    self,
    query: str,
    customer_name: Optional[str] = None,
    customer_id: Optional[str] = None,
    top_n: int = 5,
    include_reasoning: bool = True,
) -> RecommendationResponse:
    """Get personalized recommendations using multi-agent workflow with tracing"""
    start_time = time.time()
    agent_execution = []

    # Generate session ID for tracing
    session_id = str(uuid.uuid4())
    user_id = customer_name or customer_id

    # Create root trace
    with trace_span(
        name="personalized_recommendation",
        user_id=user_id,
        metadata={
            "query": query,
            "top_n": top_n,
            "include_reasoning": include_reasoning
        }
    ) as trace:

        try:
            # STEP 0: Classify query intent
            with trace_span(
                name="intent_classification",
                user_id=user_id,
                metadata={"query": query}
            ):
                logger.info(f"[Intent Classification Agent] Analyzing query")
                intent_result = self.intent_classifier_agent.classify(query)
                # ... rest of logic

            # STEP 1: Customer Profiling
            with trace_span(
                name="customer_profiling",
                user_id=user_id,
                metadata={"customer_id": customer_id}
            ):
                profile = await self.customer_service.get_customer_profile(customer_id)

            # ... rest of the workflow with trace spans

            if trace:
                trace.update(
                    output=response.model_dump(),
                    metadata={
                        "processing_time_ms": response.processing_time_ms,
                        "recommendations_count": len(response.recommendations),
                        "intent": intent.value,
                        "confidence": confidence
                    }
                )

            return response

        except Exception as e:
            if trace:
                trace.update(
                    status="error",
                    metadata={"error": str(e)}
                )
            raise
        finally:
            # Flush traces
            LangFuseTracer.flush()
```

### 7. API Endpoint Integration

Update [app/api/v1/endpoints/recommendations.py](personalized-shopping-api/app/api/v1/endpoints/recommendations.py):

```python
@router.post("/personalized")
async def get_personalized_recommendations(
    request: RecommendationRequest,
    service: Annotated[RecommendationService, Depends(get_recommendation_service)],
    background_tasks: BackgroundTasks,
    request_id: str = Header(None, alias="X-Request-ID"),
) -> RecommendationResponse:
    """Get personalized recommendations for a customer with full tracing"""

    # Generate request ID if not provided
    if not request_id:
        request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Processing recommendation request for customer: {request.customer_name or request.customer_id}"
        )

        # Execute multi-agent workflow with tracing
        response = await service.get_personalized_recommendations(
            query=request.query,
            customer_name=request.customer_name,
            customer_id=request.customer_id,
            top_n=request.top_n,
            include_reasoning=request.include_reasoning,
        )

        # Log success to LangFuse
        from app.core.tracing import LangFuseTracer
        client = LangFuseTracer.get_client()
        if client:
            client.score(
                trace_id=request_id,
                name="api_success",
                value=1.0,
                comment=f"Successfully processed: {len(response.recommendations)} recommendations"
            )

        return response

    except Exception as e:
        logger.error(f"Recommendation processing failed: {e}", exc_info=True)

        # Log failure to LangFuse
        from app.core.tracing import LangFuseTracer
        client = LangFuseTracer.get_client()
        if client:
            client.score(
                trace_id=request_id,
                name="api_error",
                value=0.0,
                comment=f"Error: {str(e)}"
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process recommendation request",
        )
```

---

## Usage Examples

### 1. Basic Tracing

```python
from app.core.tracing import trace_span

# Automatic tracing of function
with trace_span("customer_lookup", user_id="887"):
    customer = get_customer(customer_id="887")
```

### 2. LLM Call Tracing

```python
from app.models.llm_factory import get_llm, LLMType

# Get LLM with automatic tracing
llm = get_llm(
    LLMType.RESPONSE,
    session_id="session-123",
    user_id="Kenneth Martinez",
    metadata={"query_type": "informational"}
)

response = llm.invoke(messages)
```

### 3. Manual Event Logging

```python
from app.core.tracing import LangFuseTracer

client = LangFuseTracer.get_client()
if client:
    # Log custom event
    client.event(
        name="product_filter_applied",
        metadata={
            "sentiment_threshold": 0.6,
            "products_filtered": 15,
            "products_remaining": 42
        }
    )
```

### 4. User Feedback Capture

```python
from app.core.tracing import LangFuseTracer

# User provides feedback on recommendation
client = LangFuseTracer.get_client()
if client:
    client.score(
        trace_id=recommendation_id,
        name="user_rating",
        value=4.5,  # 1-5 stars
        comment="Great recommendations, very relevant!"
    )
```

---

## LangFuse Dashboard Features

Once integrated, you'll have access to:

### 1. **Traces View**
- Complete request/response traces
- LLM call details (prompts, completions, tokens)
- Latency breakdown by agent/component
- Error tracking and debugging

### 2. **Analytics**
- Query intent distribution
- Popular product categories
- User behavior patterns
- Model performance metrics

### 3. **Cost Tracking**
- Token usage per request
- Cost per user/session
- Model-specific costs
- Cost trends over time

### 4. **Prompt Management**
- Version control for prompts
- A/B testing different prompts
- Prompt performance analytics
- Rollback capabilities

### 5. **Evaluation**
- Automated quality checks
- Custom evaluation metrics
- Benchmark comparisons
- Regression detection

---

## Best Practices

### 1. **Trace Granularity**
- ✅ Trace each agent in the workflow
- ✅ Trace LLM calls separately
- ✅ Include meaningful metadata
- ❌ Don't trace every database query

### 2. **Metadata**
Always include:
- `user_id`: Customer identifier
- `session_id`: Request/conversation ID
- `query_type`: informational/recommendation
- `intent_confidence`: Classification confidence
- `products_returned`: Number of recommendations

### 3. **Tags**
Use consistent tags:
- Agent names: `intent_classifier`, `customer_profiling`
- Model types: `llama3.1:8b`, `bge-base`
- Intent types: `informational`, `recommendation`
- Categories: `total_purchases`, `spending`, etc.

### 4. **Error Handling**
```python
try:
    with trace_span("agent_execution") as trace:
        result = await execute_agent()
        trace.update(output=result)
except Exception as e:
    if trace:
        trace.update(
            status="error",
            metadata={"error": str(e), "error_type": type(e).__name__}
        )
    raise
```

### 5. **Performance**
- Use `LangFuseTracer.flush()` strategically
- Batch traces in background tasks
- Set appropriate timeouts
- Monitor LangFuse API latency

---

## Self-Hosting LangFuse

For production environments, consider self-hosting:

```yaml
# docker-compose.yml addition
langfuse:
  image: langfuse/langfuse:latest
  ports:
    - "3000:3000"
  environment:
    DATABASE_URL: postgresql://user:pass@db:5432/langfuse
    NEXTAUTH_SECRET: your-secret-key
    SALT: your-salt-key
  depends_on:
    - db
  networks:
    - shopping-network
```

Then update `.env`:
```bash
LANGFUSE_HOST=http://localhost:3000
```

---

## Migration from LangSmith

To migrate from LangSmith to LangFuse:

1. **Keep both enabled** during transition:
   ```bash
   ENABLE_TRACING=true
   LANGSMITH_API_KEY=your-key  # Keep for now
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=your-key
   ```

2. **Compare traces** side-by-side for 1-2 weeks

3. **Disable LangSmith** once satisfied:
   ```bash
   LANGSMITH_API_KEY=  # Remove
   ```

4. **Remove dependency** from requirements.txt

---

## Troubleshooting

### LangFuse Not Receiving Traces
- Check `LANGFUSE_ENABLED=true` in .env
- Verify API keys are correct
- Check network connectivity
- Review logs for initialization errors

### High Latency
- Traces are sent asynchronously
- Use `flush()` only when necessary
- Consider batching in background tasks
- Monitor LangFuse host response times

### Missing Metadata
- Ensure callbacks are passed to LLM calls
- Verify session_id/user_id are set
- Check trace span nesting is correct

---

## Summary

**LangFuse integration provides:**
- ✅ Complete observability for LLM-based agents
- ✅ Cost tracking and optimization insights
- ✅ User behavior analytics
- ✅ Prompt version management
- ✅ Quality evaluation tools
- ✅ Self-hosting option for data privacy

**Next Steps:**
1. Add LangFuse dependencies
2. Configure environment variables
3. Implement core tracing module
4. Update LLM factory and agents
5. Test in development environment
6. Enable in production with monitoring
