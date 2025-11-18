"""
Personalized Recommendation Workflow

Orchestrates the multi-agent recommendation pipeline from customer profiling
through response generation. This workflow is the production path for the
/recommendations/personalized endpoint.
"""

from typing import List, Optional
from collections import Counter
import pandas as pd
import time
import logging

from app.capabilities.base import AgentContext
from app.capabilities.agents.customer_profiling import (
    CustomerProfilingAgent,
    CustomerProfilingInput,
)
from app.capabilities.agents.similar_customers import (
    SimilarCustomersAgent,
    SimilarCustomersInput,
)
from app.capabilities.agents.sentiment_filtering import (
    SentimentFilteringAgent,
    SentimentFilteringInput,
    ProductCandidate,
)
from app.capabilities.agents.product_scoring import (
    ProductScoringAgent,
    ProductScoringInput,
    ScoredProduct,
)
from app.capabilities.agents.response_generation import (
    ResponseGenerationAgent,
    ResponseGenerationInput,
)

from app.domain.schemas.recommendation import RecommendationResponse
from app.domain.schemas.customer import CustomerProfileSummary, SimilarCustomer
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository
from app.repositories.review_repository import ReviewRepository
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.core.config import settings

logger = logging.getLogger(__name__)


class PersonalizedRecommendationWorkflow:
    """
    Multi-agent workflow for personalized product recommendations

    This workflow orchestrates 5 agents in sequence:
    1. Customer Profiling - Extract behavioral metrics from purchase history
    2. Similar Customer Discovery - Find customers with similar behavior
    3. Sentiment Filtering - Remove products with poor reviews
    4. Product Scoring - Rank products using collaborative filtering + category affinity
    5. Response Generation - Generate natural language explanation

    The workflow is stateless and coordinates agents via Pydantic models.
    All business logic lives in the agents themselves.
    """

    def __init__(
        self,
        customer_repository: CustomerRepository,
        vector_repository: VectorRepository,
        review_repository: ReviewRepository,
        sentiment_analyzer: SentimentAnalyzer,
    ):
        """
        Initialize the workflow with dependencies

        Args:
            customer_repository: Customer data access
            vector_repository: Vector similarity search
            review_repository: Review data access
            sentiment_analyzer: Sentiment analysis model
        """
        # Initialize agents
        self.profiling_agent = CustomerProfilingAgent(customer_repository)
        self.similar_customers_agent = SimilarCustomersAgent(
            customer_repository,
            vector_repository
        )
        self.sentiment_filtering_agent = SentimentFilteringAgent(
            review_repository,
            sentiment_analyzer
        )
        self.product_scoring_agent = ProductScoringAgent()
        self.response_generation_agent = ResponseGenerationAgent()

        # Store repositories for data access between agents
        self.customer_repo = customer_repository

    async def execute(
        self,
        customer_id: str,
        query: str,
        top_n: int = 5,
        include_reasoning: bool = True,
        context: Optional[AgentContext] = None,
    ) -> RecommendationResponse:
        """
        Execute the personalized recommendation workflow

        Args:
            customer_id: Customer to recommend for
            query: User's search query
            top_n: Number of recommendations to return
            include_reasoning: Whether to generate LLM reasoning
            context: Execution context (created if not provided)

        Returns:
            Complete recommendation response with products and reasoning
        """
        # Create context if not provided
        if context is None:
            import uuid
            context = AgentContext(
                request_id=f"rec-{uuid.uuid4().hex[:8]}",
                session_id=None,
                user_id=customer_id,
                metadata={},
            )

        start_time = time.time()
        agent_execution = []

        logger.info(
            f"Starting personalized recommendation workflow for customer {customer_id}"
        )

        # ===================================================================
        # AGENT 1: Customer Profiling
        # ===================================================================
        agent_execution.append("customer_profiling")
        profiling_input = CustomerProfilingInput(customer_id=customer_id)
        profiling_output = await self.profiling_agent.run(profiling_input, context)
        profile = profiling_output.profile

        logger.info(
            f"[Agent 1] Profiled customer: {profile.total_purchases} purchases, "
            f"{profile.purchase_frequency} frequency"
        )

        # ===================================================================
        # AGENT 2: Similar Customer Discovery
        # ===================================================================
        agent_execution.append("similar_customer_discovery")
        similar_input = SimilarCustomersInput(
            customer_profile=profile,
            top_k=settings.SIMILARITY_TOP_K,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
        )
        similar_output = await self.similar_customers_agent.run(similar_input, context)
        similar_customers = similar_output.similar_customers

        logger.info(
            f"[Agent 2] Found {len(similar_customers)} similar customers"
        )

        # ===================================================================
        # Data Collection: Gather candidate products from similar customers
        # ===================================================================
        candidate_purchases = []
        for sim_customer in similar_customers:
            purchases = self.customer_repo.get_purchases_by_customer_id(
                sim_customer.customer_id
            )
            candidate_purchases.extend(purchases)

        # Get customer's already purchased products to exclude
        customer_purchases = self.customer_repo.get_purchases_by_customer_id(customer_id)
        already_purchased_ids = set(str(p["product_id"]) for p in customer_purchases)

        # Filter out already purchased
        candidate_purchases = [
            p for p in candidate_purchases
            if str(p["product_id"]) not in already_purchased_ids
        ]

        logger.info(
            f"[Data] Collected {len(candidate_purchases)} candidate purchases "
            f"(excluded {len(already_purchased_ids)} already purchased)"
        )

        # Aggregate products
        if not candidate_purchases:
            # No candidates - return empty response
            logger.warning("No candidate products found")
            return self._build_empty_response(
                query,
                profile,
                agent_execution,
                time.time() - start_time
            )

        df = pd.DataFrame(candidate_purchases)
        products_df = df.groupby("product_id").agg({
            "product_name": "first",
            "product_category": "first",
            "price": "mean",
            "transaction_id": "count",
        }).reset_index()
        products_df.columns = [
            "product_id",
            "product_name",
            "product_category",
            "avg_price",
            "purchase_count"
        ]
        candidate_products = products_df.to_dict("records")

        # ===================================================================
        # AGENT 3: Sentiment Filtering
        # ===================================================================
        agent_execution.append("sentiment_filtering")
        sentiment_input = SentimentFilteringInput(
            candidate_products=[
                ProductCandidate(
                    product_id=str(p["product_id"]),
                    product_name=p["product_name"],
                    product_category=p["product_category"],
                    avg_price=float(p["avg_price"]),
                    purchase_count=int(p["purchase_count"]),
                )
                for p in candidate_products
            ],
            sentiment_threshold=settings.SENTIMENT_THRESHOLD,
        )
        sentiment_output = await self.sentiment_filtering_agent.run(
            sentiment_input,
            context
        )
        filtered_products = sentiment_output.filtered_products

        logger.info(
            f"[Agent 3] Filtered to {len(filtered_products)} products "
            f"(removed {sentiment_output.products_filtered_out})"
        )

        if not filtered_products:
            logger.warning("All products filtered out by sentiment")
            return self._build_empty_response(
                query,
                profile,
                agent_execution,
                time.time() - start_time
            )

        # ===================================================================
        # AGENT 4: Product Scoring and Ranking
        # ===================================================================
        agent_execution.append("product_scoring")

        # Build purchase counts for collaborative filtering
        product_purchase_counts = Counter(
            str(p["product_id"]) for p in candidate_purchases
        )

        scoring_input = ProductScoringInput(
            customer_profile=profile,
            products=[
                ScoredProduct(
                    product_id=p.product_id,
                    product_name=p.product_name,
                    product_category=p.product_category,
                    avg_price=p.avg_price,
                    purchase_count=p.purchase_count,
                    avg_sentiment=p.avg_sentiment,
                    review_count=p.review_count,
                )
                for p in filtered_products
            ],
            purchase_counts=product_purchase_counts,
            top_n=top_n,
            max_per_category=2,  # Diversity constraint
        )
        scoring_output = await self.product_scoring_agent.run(scoring_input, context)
        recommendations = scoring_output.recommendations

        logger.info(
            f"[Agent 4] Generated {len(recommendations)} recommendations"
        )

        # ===================================================================
        # AGENT 5: Response Generation
        # ===================================================================
        agent_execution.append("response_generation")

        if include_reasoning and recommendations:
            response_input = ResponseGenerationInput(
                query=query,
                customer_profile=profile,
                recommendations=recommendations,
            )
            response_output = await self.response_generation_agent.run(
                response_input,
                context
            )
            reasoning = response_output.reasoning
        else:
            reasoning = f"Based on {profile.customer_name}'s purchase history, here are {len(recommendations)} recommendations."

        logger.info(
            f"[Agent 5] Generated reasoning ({len(reasoning)} chars)"
        )

        # ===================================================================
        # Build Final Response
        # ===================================================================
        processing_time_ms = (time.time() - start_time) * 1000

        response = RecommendationResponse(
            query=query,
            customer_profile=CustomerProfileSummary(
                customer_id=profile.customer_id,
                customer_name=profile.customer_name,
                total_purchases=profile.total_purchases,
                avg_purchase_price=profile.avg_purchase_price,
                favorite_categories=profile.favorite_categories,
                price_segment=profile.price_segment,
                purchase_frequency=profile.purchase_frequency,
            ),
            recommendations=recommendations,
            reasoning=reasoning,
            confidence_score=min(len(similar_customers) / 20.0, 1.0),
            processing_time_ms=processing_time_ms,
            similar_customers_analyzed=len(similar_customers),
            products_considered=len(candidate_products),
            products_filtered_by_sentiment=sentiment_output.products_filtered_out,
            recommendation_strategy="collaborative_with_category_affinity",
            agent_execution_order=agent_execution,
            metadata={
                "fallback_used": False,
                "session_id": context.session_id,
                "request_id": context.request_id,
            },
        )

        logger.info(
            f"Workflow completed in {processing_time_ms:.0f}ms: "
            f"{len(recommendations)} recommendations"
        )

        return response

    def _build_empty_response(
        self,
        query: str,
        profile,
        agent_execution: List[str],
        elapsed_time: float,
    ) -> RecommendationResponse:
        """Build empty response when no recommendations found"""
        return RecommendationResponse(
            query=query,
            customer_profile=CustomerProfileSummary(
                customer_id=profile.customer_id,
                customer_name=profile.customer_name,
                total_purchases=profile.total_purchases,
                avg_purchase_price=profile.avg_purchase_price,
                favorite_categories=profile.favorite_categories,
                price_segment=profile.price_segment,
                purchase_frequency=profile.purchase_frequency,
            ),
            recommendations=[],
            reasoning=f"Unfortunately, no suitable recommendations were found for {profile.customer_name} at this time.",
            confidence_score=0.0,
            processing_time_ms=elapsed_time * 1000,
            similar_customers_analyzed=0,
            products_considered=0,
            products_filtered_by_sentiment=0,
            recommendation_strategy="collaborative_with_category_affinity",
            agent_execution_order=agent_execution,
            metadata={"fallback_used": False},
        )
