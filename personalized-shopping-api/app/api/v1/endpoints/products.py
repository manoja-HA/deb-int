"""
Product API endpoints
"""

from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.api.dependencies import get_product_service
from app.domain.schemas.product import ProductReview
from app.services.product_service import ProductService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["products"])

@router.get(
    "/{product_id}/reviews",
    response_model=List[ProductReview],
    status_code=status.HTTP_200_OK,
    summary="Get product reviews with sentiment analysis",
    description="Get all reviews for a product with computed sentiment scores",
)
async def get_product_reviews(
    product_id: str,
    service: Annotated[ProductService, Depends(get_product_service)],
) -> List[ProductReview]:
    """Get product reviews with sentiment"""
    try:
        reviews = await service.get_product_reviews_with_sentiment(product_id)
        return reviews

    except Exception as e:
        logger.error(f"Failed to get reviews: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product reviews",
        )
