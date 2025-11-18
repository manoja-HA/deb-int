"""
Similar Customers Discovery Agent

Finds customers with similar purchase behavior using vector similarity search.
This agent encapsulates vector search logic and similarity computation.
"""

from typing import List
from pydantic import BaseModel, Field
import logging

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.domain.schemas.customer import CustomerProfile, SimilarCustomer
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository
from app.core.config import settings

logger = logging.getLogger(__name__)


class SimilarCustomersInput(BaseModel):
    """Input for similar customers agent"""

    customer_profile: CustomerProfile = Field(
        description="Profile of the customer to find similar customers for"
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of similar customers to return"
    )
    similarity_threshold: float = Field(
        default=settings.SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )


class SimilarCustomersOutput(BaseModel):
    """Output from similar customers agent"""

    similar_customers: List[SimilarCustomer] = Field(
        description="List of similar customers with similarity scores"
    )
    search_query: str = Field(
        description="The behavior text used for similarity search"
    )


class SimilarCustomersAgent(BaseAgent[SimilarCustomersInput, SimilarCustomersOutput]):
    """
    Agent that discovers similar customers via vector similarity

    Responsibilities:
    - Convert customer profile to behavior text representation
    - Perform vector similarity search in ChromaDB
    - Retrieve metadata for similar customers
    - Calculate common category overlap
    - Filter by similarity threshold

    This agent encapsulates the similar customer discovery logic that was
    previously in CustomerService.get_similar_customers().
    """

    def __init__(
        self,
        customer_repository: CustomerRepository,
        vector_repository: VectorRepository,
    ):
        """
        Initialize the similar customers agent

        Args:
            customer_repository: Repository for customer data access
            vector_repository: Repository for vector similarity search
        """
        metadata = AgentMetadata(
            id="similar_customers",
            name="Similar Customers Discovery Agent",
            description="Finds customers with similar purchase behavior using vector search",
            version="1.0.0",
            input_schema=SimilarCustomersInput,
            output_schema=SimilarCustomersOutput,
            tags=["customer", "similarity", "vector-search", "collaborative-filtering"],
        )
        super().__init__(metadata)
        self.customer_repo = customer_repository
        self.vector_repo = vector_repository

    async def _execute(
        self,
        input_data: SimilarCustomersInput,
        context: AgentContext,
    ) -> SimilarCustomersOutput:
        """
        Find similar customers using vector similarity

        Args:
            input_data: Contains customer profile and search parameters
            context: Execution context

        Returns:
            List of similar customers with similarity scores and common categories
        """
        profile = input_data.customer_profile
        top_k = input_data.top_k
        threshold = input_data.similarity_threshold

        # Create behavior text for embedding
        behavior_text = self._create_behavior_text(profile)

        self._logger.debug(
            f"Searching for similar customers with query: {behavior_text}"
        )

        # Perform vector similarity search
        # ChromaDB will handle embedding generation internally
        similar_results = self.vector_repo.search_similar(
            query_text=behavior_text,
            top_k=top_k + 1,  # +1 to account for potential self-match
            threshold=threshold,
        )

        # Build similar customer objects
        similar_customers = []
        for sim_customer_id, similarity in similar_results:
            # Skip self-match
            if sim_customer_id == profile.customer_id:
                continue

            # Get purchase data for this similar customer
            their_purchases = self.customer_repo.get_purchases_by_customer_id(
                sim_customer_id
            )
            if not their_purchases:
                continue

            # Calculate category overlap
            their_categories = list(
                set(p["product_category"] for p in their_purchases)
            )
            common_categories = list(
                set(profile.favorite_categories) & set(their_categories)
            )

            # Get metadata from vector store
            metadata = self.vector_repo.get_metadata(sim_customer_id) or {}

            # Build similar customer object
            similar_customer = SimilarCustomer(
                customer_id=sim_customer_id,
                customer_name=metadata.get(
                    "customer_name",
                    f"Customer {sim_customer_id}"
                ),
                similarity_score=similarity,
                common_categories=common_categories,
                purchase_overlap_count=len(common_categories),
            )

            similar_customers.append(similar_customer)

        # Limit to requested top_k (after removing self-match)
        similar_customers = similar_customers[:top_k]

        self._logger.info(
            f"Found {len(similar_customers)} similar customers for {profile.customer_id}"
        )

        return SimilarCustomersOutput(
            similar_customers=similar_customers,
            search_query=behavior_text,
        )

    def _create_behavior_text(self, profile: CustomerProfile) -> str:
        """
        Create text representation of customer behavior for embedding

        This text representation captures the key behavioral signals:
        - Purchase frequency
        - Price segment
        - Favorite categories
        - Average purchase price

        Args:
            profile: Customer profile

        Returns:
            Text representation suitable for embedding
        """
        categories = ", ".join(profile.favorite_categories)
        return (
            f"Frequency: {profile.purchase_frequency} buyer, "
            f"Price: {profile.price_segment}, "
            f"Categories: {categories}, "
            f"Avg: ${profile.avg_purchase_price:.2f}"
        )
