"""Domain schemas module"""

from .base import TimestampMixin, PaginationParams
from .customer import CustomerProfile, CustomerProfileSummary
from .product import ProductRecommendation, ProductReview
from .recommendation import RecommendationRequest, RecommendationResponse

__all__ = [
    "TimestampMixin",
    "PaginationParams",
    "CustomerProfile",
    "CustomerProfileSummary",
    "ProductRecommendation",
    "ProductReview",
    "RecommendationRequest",
    "RecommendationResponse",
]
