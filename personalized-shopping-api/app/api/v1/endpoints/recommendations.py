"""
Recommendation API endpoints
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
import logging

from app.api.dependencies import get_recommendation_service
from app.domain.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationService
from app.core.exceptions import NotFoundException, ValidationException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.post(
    "/personalized",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get personalized product recommendations",
    description="""
    Generate personalized product recommendations using multi-agent AI system.

    The system:
    1. Profiles the customer from purchase history
    2. Finds similar customers using vector similarity
    3. Filters products by review sentiment
    4. Generates cross-category recommendations
    5. Returns natural language response with products
    """,
)
async def get_personalized_recommendations(
    request: RecommendationRequest,
    service: Annotated[RecommendationService, Depends(get_recommendation_service)],
    background_tasks: BackgroundTasks,
) -> RecommendationResponse:
    """Get personalized recommendations for a customer"""
    try:
        logger.info(
            f"Processing recommendation request for customer: {request.customer_name or request.customer_id}"
        )

        # Execute multi-agent workflow
        response = await service.get_personalized_recommendations(
            query=request.query,
            customer_name=request.customer_name,
            customer_id=request.customer_id,
            top_n=request.top_n,
            include_reasoning=request.include_reasoning,
        )

        # Log metrics in background
        background_tasks.add_task(
            log_recommendation_metrics,
            response=response,
            customer_id=request.customer_id or request.customer_name,
        )

        logger.info(
            f"Recommendation completed: {len(response.recommendations)} products, "
            f"confidence: {response.confidence_score:.2f}"
        )

        return response

    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Recommendation processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process recommendation request",
        )

async def log_recommendation_metrics(response: RecommendationResponse, customer_id: str):
    """Log recommendation metrics for monitoring"""
    logger.info(
        "Recommendation metrics",
        extra={
            "customer_id": customer_id,
            "recommendations_count": len(response.recommendations),
            "confidence": response.confidence_score,
            "processing_time_ms": response.processing_time_ms,
        },
    )
