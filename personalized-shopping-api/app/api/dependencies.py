"""
FastAPI dependencies for dependency injection
"""

from typing import Annotated
from fastapi import Depends

from app.services.recommendation_service import RecommendationService
from app.services.customer_service import CustomerService
from app.services.product_service import ProductService
from app.repositories.customer_repository import CustomerRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.review_repository import ReviewRepository
from app.repositories.vector_repository import VectorRepository

# ===== Repository Dependencies =====

def get_customer_repository() -> CustomerRepository:
    """Get customer repository instance"""
    return CustomerRepository()

def get_product_repository() -> ProductRepository:
    """Get product repository instance"""
    return ProductRepository()

def get_review_repository() -> ReviewRepository:
    """Get review repository instance"""
    return ReviewRepository()

def get_vector_repository() -> VectorRepository:
    """Get vector repository instance (singleton)"""
    return VectorRepository.get_instance()

# ===== Service Dependencies =====

def get_customer_service(
    customer_repo: Annotated[CustomerRepository, Depends(get_customer_repository)],
    vector_repo: Annotated[VectorRepository, Depends(get_vector_repository)],
) -> CustomerService:
    """Get customer service instance"""
    return CustomerService(
        customer_repository=customer_repo,
        vector_repository=vector_repo,
    )

def get_product_service(
    product_repo: Annotated[ProductRepository, Depends(get_product_repository)],
    review_repo: Annotated[ReviewRepository, Depends(get_review_repository)],
) -> ProductService:
    """Get product service instance"""
    return ProductService(
        product_repository=product_repo,
        review_repository=review_repo,
    )

def get_recommendation_service(
    customer_service: Annotated[CustomerService, Depends(get_customer_service)],
    product_service: Annotated[ProductService, Depends(get_product_service)],
    vector_repo: Annotated[VectorRepository, Depends(get_vector_repository)],
) -> RecommendationService:
    """Get recommendation service instance"""
    return RecommendationService(
        customer_service=customer_service,
        product_service=product_service,
        vector_repository=vector_repo,
    )
