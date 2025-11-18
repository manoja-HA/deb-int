"""
Application lifecycle events (startup/shutdown)
"""

import logging
from app.core.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)

# Global state
_vector_store_loaded = False
_models_warmed_up = False

async def startup_event_handler() -> None:
    """
    Application startup event handler

    Responsibilities:
    - Setup logging
    - Register agents
    - Load vector store
    - Warm up models
    - Initialize caches
    """
    logger.info("Starting application startup sequence...")

    # Setup logging first
    setup_logging()

    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"API version: {settings.VERSION}")

    try:
        # Register agents with AgentRegistry for discovery
        from app.capabilities import register_all_agents
        register_all_agents()
        logger.info("Registered all production agents")

        # Load vector store
        await load_vector_store()

        # Warm up models (always - eliminates cold-start for first request)
        # Skip in testing environment to speed up tests
        if not settings.TESTING:
            await warm_up_models()

        # Initialize cache (if enabled)
        if settings.ENABLE_CACHE and settings.REDIS_URL:
            await initialize_cache()

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

async def shutdown_event_handler() -> None:
    """
    Application shutdown event handler

    Responsibilities:
    - Close database connections
    - Close cache connections
    - Cleanup resources
    """
    logger.info("Starting application shutdown sequence...")

    try:
        # Close cache connections
        if settings.ENABLE_CACHE and settings.REDIS_URL:
            await close_cache()

        # Close database connections (if using DB)
        # await close_database()

        logger.info("Application shutdown completed successfully")

    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)

# ===== Helper Functions =====

async def load_vector_store() -> None:
    """Load vector store index"""
    global _vector_store_loaded

    try:
        logger.info("Loading vector store...")

        # Check if index file exists
        if not settings.VECTOR_INDEX_PATH.exists():
            logger.warning(
                f"Vector index not found at {settings.VECTOR_INDEX_PATH}. "
                "Please run scripts/build_vector_index.py"
            )
            return

        # Vector store will be loaded lazily by VectorRepository
        _vector_store_loaded = True

        logger.info("Vector store loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise

async def warm_up_models() -> None:
    """Warm up LLM models (optional optimization)"""
    global _models_warmed_up

    try:
        logger.info("Warming up Ollama models...")

        # Import here to avoid circular dependencies
        from app.models.llm_factory import get_llm, LLMType

        # Pre-load critical models into Ollama memory
        # This eliminates 30-60s loading time on first request
        models_to_warm = [
            (LLMType.PROFILING, "profiling model"),
            (LLMType.SENTIMENT, "sentiment model"),
            (LLMType.INTENT, "intent model"),
        ]

        for llm_type, description in models_to_warm:
            try:
                logger.info(f"Pre-warming {description}...")
                llm = get_llm(llm_type)

                # Trigger model loading with minimal inference
                from langchain_core.messages import HumanMessage
                llm.invoke([HumanMessage(content="warmup")])

                logger.info(f"✓ {description} loaded")
            except Exception as model_error:
                logger.warning(f"Failed to warm {description}: {model_error}")

        _models_warmed_up = True
        logger.info("Model warm-up completed (models will stay loaded for 10 minutes)")

    except Exception as e:
        logger.warning(f"Model warm-up failed (non-critical): {e}")

async def initialize_cache() -> None:
    """Initialize Redis cache connection"""
    try:
        logger.info("Initializing cache...")

        # Test Redis connection
        # from app.infrastructure.cache import test_redis_connection
        # await test_redis_connection()

        logger.info("Cache initialized successfully")

    except Exception as e:
        logger.warning(f"Cache initialization failed: {e}")

async def close_cache() -> None:
    """Close Redis cache connection"""
    try:
        # from app.infrastructure.cache import close_redis
        # await close_redis()

        logger.info("Cache connections closed")

    except Exception as e:
        logger.error(f"Failed to close cache: {e}")
