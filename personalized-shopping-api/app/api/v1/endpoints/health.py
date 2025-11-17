"""
Health check endpoints
"""

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    environment: str
    version: str

@router.get("", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
    )
