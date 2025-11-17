"""
FastAPI application entry point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.router import api_router as api_v1_router
from app.core.config import settings
from app.core.exceptions import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from app.core.events import startup_event_handler, shutdown_event_handler
from app.core.logging import setup_logging

# Setup logging first
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await startup_event_handler()
    yield
    # Shutdown
    await shutdown_event_handler()

# Initialize FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multi-Agent Personalized Shopping Assistant API",
    docs_url=f"{settings.API_V1_PREFIX}/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url=f"{settings.API_V1_PREFIX}/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# CORS Middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(422, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API routers
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

# Prometheus metrics
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Health check
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION,
    }

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Personalized Shopping Assistant API",
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": "/health",
        "version": settings.VERSION,
    }
