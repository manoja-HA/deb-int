"""
Customer API endpoints
"""

from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.api.dependencies import get_customer_service
from app.domain.schemas.customer import CustomerProfile, SimilarCustomer
from app.services.customer_service import CustomerService
from app.core.exceptions import NotFoundException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customers", tags=["customers"])

@router.get(
    "/{customer_id}/profile",
    response_model=CustomerProfile,
    status_code=status.HTTP_200_OK,
    summary="Get customer profile",
    description="Get detailed customer profile with purchase history and preferences",
)
async def get_customer_profile(
    customer_id: str,
    service: Annotated[CustomerService, Depends(get_customer_service)],
) -> CustomerProfile:
    """Get customer profile by ID"""
    try:
        profile = await service.get_customer_profile(customer_id)
        return profile

    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@router.get(
    "/{customer_id}/similar",
    response_model=List[SimilarCustomer],
    status_code=status.HTTP_200_OK,
    summary="Find similar customers",
    description="Find customers with similar purchase behaviors using vector similarity",
)
async def get_similar_customers(
    customer_id: str,
    service: Annotated[CustomerService, Depends(get_customer_service)],
    top_k: int = 20,
) -> List[SimilarCustomer]:
    """Find similar customers"""
    try:
        similar = await service.get_similar_customers(customer_id, top_k=top_k)
        return similar

    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
