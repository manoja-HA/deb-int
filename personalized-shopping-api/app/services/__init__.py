"""Service layer module"""

from .customer_service import CustomerService
from .product_service import ProductService
from .recommendation_service import RecommendationService

__all__ = [
    "CustomerService",
    "ProductService",
    "RecommendationService",
]
