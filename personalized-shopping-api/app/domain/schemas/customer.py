"""
Customer domain schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Purchase(BaseModel):
    """Single purchase record"""

    transaction_id: str
    product_id: str
    product_name: str
    product_category: str
    quantity: int = Field(ge=1)
    price: float = Field(ge=0)
    purchase_date: str
    country: str

    model_config = ConfigDict(from_attributes=True)

class CustomerProfileSummary(BaseModel):
    """Lightweight customer profile for API responses"""

    customer_id: str
    customer_name: str
    total_purchases: int = Field(ge=0)
    avg_purchase_price: float = Field(ge=0)
    favorite_categories: List[str]
    price_segment: str = Field(examples=["budget", "mid-range", "premium"])
    purchase_frequency: Optional[str] = Field(
        default=None,
        examples=["low", "medium", "high"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "customer_id": "887",
                    "customer_name": "Kenneth Martinez",
                    "total_purchases": 5,
                    "avg_purchase_price": 689.99,
                    "favorite_categories": ["Electronics"],
                    "price_segment": "premium",
                    "purchase_frequency": "low",
                }
            ]
        }
    )

class CustomerProfile(CustomerProfileSummary):
    """Complete customer profile with purchase history"""

    total_spent: float = Field(ge=0)
    favorite_brands: List[str] = Field(default_factory=list)
    recent_purchases: List[Purchase] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="Profile confidence score")

class SimilarCustomer(BaseModel):
    """Similar customer with similarity score"""

    customer_id: str
    customer_name: str
    similarity_score: float = Field(ge=0, le=1)
    common_categories: List[str]
    purchase_overlap_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "customer_id": "560",
                    "customer_name": "Jane Smith",
                    "similarity_score": 0.89,
                    "common_categories": ["Electronics", "Books"],
                    "purchase_overlap_count": 3,
                }
            ]
        }
    )
