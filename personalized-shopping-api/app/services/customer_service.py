"""Customer service - Business logic for customer operations"""

from typing import List, Optional
import logging

from app.domain.schemas.customer import CustomerProfile, SimilarCustomer, Purchase
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository
from app.core.exceptions import NotFoundException
from app.models.embedding_model import get_embedding_model
from app.core.config import settings

logger = logging.getLogger(__name__)

class CustomerService:
    """Customer business logic service"""

    def __init__(
        self,
        customer_repository: CustomerRepository,
        vector_repository: VectorRepository,
    ):
        self.customer_repo = customer_repository
        self.vector_repo = vector_repository

    async def get_customer_profile(self, customer_id: str) -> CustomerProfile:
        """Get complete customer profile"""
        customer = self.customer_repo.get_by_id(customer_id)

        if not customer:
            raise NotFoundException("Customer", customer_id)

        # Get purchases
        purchases = self.customer_repo.get_purchases_by_customer_id(customer_id)

        if not purchases:
            raise NotFoundException("Customer purchases", customer_id)

        # Calculate metrics
        import pandas as pd
        df = pd.DataFrame(purchases)

        total_purchases = len(purchases)
        total_spent = float(df["price"].sum())
        avg_price = float(df["price"].mean())

        # Get favorite categories
        category_counts = df["product_category"].value_counts()
        favorite_categories = category_counts.head(3).index.tolist()

        # Determine segments
        if total_purchases > 10:
            frequency = "high"
        elif total_purchases >= 3:
            frequency = "medium"
        else:
            frequency = "low"

        if avg_price < 200:
            price_segment = "budget"
        elif avg_price < 600:
            price_segment = "mid-range"
        else:
            price_segment = "premium"

        # Build profile
        profile = CustomerProfile(
            customer_id=customer_id,
            customer_name=customer["customer_name"],
            total_purchases=total_purchases,
            total_spent=total_spent,
            avg_purchase_price=avg_price,
            favorite_categories=favorite_categories,
            favorite_brands=[],
            price_segment=price_segment,
            purchase_frequency=frequency,
            recent_purchases=[Purchase(**p) for p in purchases[-5:]],
            confidence=min(total_purchases / 10.0, 1.0),
        )

        return profile

    async def get_similar_customers(
        self,
        customer_id: str,
        top_k: int = 20,
    ) -> List[SimilarCustomer]:
        """Find similar customers using vector similarity"""
        # Get customer profile
        profile = await self.get_customer_profile(customer_id)

        # Create behavior text (ChromaDB will handle embedding)
        behavior_text = self._create_behavior_text(profile)

        # Search vector store
        similar_results = self.vector_repo.search_similar(
            query_text=behavior_text,
            top_k=top_k + 1,  # +1 to exclude self
            threshold=settings.SIMILARITY_THRESHOLD,
        )

        # Build results
        similar_customers = []
        for sim_customer_id, similarity in similar_results:
            if sim_customer_id == customer_id:
                continue  # Skip self

            # Get purchases for common categories
            their_purchases = self.customer_repo.get_purchases_by_customer_id(sim_customer_id)
            if not their_purchases:
                continue

            their_categories = list(set(p["product_category"] for p in their_purchases))
            common_categories = list(set(profile.favorite_categories) & set(their_categories))

            metadata = self.vector_repo.get_metadata(sim_customer_id) or {}

            similar_customer = SimilarCustomer(
                customer_id=sim_customer_id,
                customer_name=metadata.get("customer_name", f"Customer {sim_customer_id}"),
                similarity_score=similarity,
                common_categories=common_categories,
                purchase_overlap_count=len(common_categories),
            )

            similar_customers.append(similar_customer)

        return similar_customers[:top_k]

    def _create_behavior_text(self, profile: CustomerProfile) -> str:
        """Create text representation for embedding"""
        categories = ", ".join(profile.favorite_categories)
        return f"Frequency: {profile.purchase_frequency} buyer, Price: {profile.price_segment}, Categories: {categories}, Avg: ${profile.avg_purchase_price:.2f}"
