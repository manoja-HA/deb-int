"""
Recommendation service - Thin facade over workflow orchestration

This service now acts as a thin adapter layer that:
1. Handles intent classification (informational vs recommendation)
2. Delegates to appropriate workflow or query answering service
3. Maintains tracing and observability
4. Provides backward compatibility with existing API
"""

from typing import Optional
import time
import logging
import uuid

from app.domain.schemas.recommendation import RecommendationResponse
from app.domain.schemas.customer import CustomerProfileSummary
from app.services.customer_service import CustomerService
from app.services.product_service import ProductService
from app.services.query_answering_service import QueryAnsweringService
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository
from app.repositories.review_repository import ReviewRepository
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.core.config import settings
from app.core.exceptions import NotFoundException, ValidationException
from app.agents.intent_classifier_agent_v2 import IntentClassifierAgentV2
from app.agents.intent_classifier_agent import QueryIntent  # Keep enum for compatibility
from app.core.tracing import trace_span
from app.capabilities.base import AgentContext
from app.workflows.personalized_recommendation import PersonalizedRecommendationWorkflow

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Recommendation service - Facade over workflows

    This service routes requests to the appropriate workflow based on intent:
    - INFORMATIONAL queries → QueryAnsweringService
    - RECOMMENDATION queries → PersonalizedRecommendationWorkflow

    The service is now a thin layer that:
    - Handles customer lookup by name
    - Performs intent classification
    - Routes to workflows
    - Maintains tracing/observability
    - Ensures backward compatibility
    """

    def __init__(
        self,
        customer_service: CustomerService,
        product_service: ProductService,
        vector_repository: VectorRepository,
    ):
        """
        Initialize the recommendation service

        Args:
            customer_service: Service for customer operations
            product_service: Service for product operations
            vector_repository: Repository for vector similarity search
        """
        self.customer_service = customer_service
        self.product_service = product_service
        self.vector_repo = vector_repository

        # Repositories
        self.customer_repo = CustomerRepository()
        self.review_repo = ReviewRepository()

        # Intent classification and query answering (using V2 with PydanticAI + Ollama)
        self.intent_classifier_agent = IntentClassifierAgentV2()
        self.query_answering_service = QueryAnsweringService()

        # Initialize workflow
        self.recommendation_workflow = PersonalizedRecommendationWorkflow(
            customer_repository=self.customer_repo,
            vector_repository=self.vector_repo,
            review_repository=self.review_repo,
            sentiment_analyzer=SentimentAnalyzer(method="rule_based"),
        )

    async def get_personalized_recommendations(
        self,
        query: str,
        customer_name: Optional[str] = None,
        customer_id: Optional[str] = None,
        top_n: int = 5,
        include_reasoning: bool = True,
    ) -> RecommendationResponse:
        """
        Get personalized recommendations using multi-agent workflow

        This method acts as the main entry point for the /recommendations/personalized
        endpoint. It:
        1. Validates input
        2. Classifies query intent (informational vs recommendation)
        3. Routes to appropriate handler
        4. Returns structured response

        Args:
            query: User's search/question query
            customer_name: Customer name (mutually exclusive with customer_id)
            customer_id: Customer ID (mutually exclusive with customer_name)
            top_n: Number of recommendations to return (1-20)
            include_reasoning: Whether to generate LLM-based reasoning

        Returns:
            RecommendationResponse with products and reasoning

        Raises:
            ValidationException: If neither customer_name nor customer_id provided
            NotFoundException: If customer not found
        """
        start_time = time.time()

        # Generate session ID for tracing
        session_id = f"req-{uuid.uuid4().hex[:8]}"

        # Validate input
        if not customer_name and not customer_id:
            raise ValidationException(
                "Either customer_name or customer_id must be provided"
            )

        # Main trace span for the entire request
        with trace_span(
            name="recommendation_workflow",
            session_id=session_id,
            user_id=customer_name or customer_id,
            metadata={
                "query": query,
                "top_n": top_n,
                "include_reasoning": include_reasoning,
            },
            tags=["recommendation", "workflow"],
            input_data={
                "query": query,
                "customer_name": customer_name,
                "customer_id": customer_id,
                "top_n": top_n,
            }
        ) as main_trace:
            try:
                # ============================================================
                # STEP 0: Intent Classification
                # ============================================================
                logger.info(
                    f"[Intent Classification] Analyzing query: '{query[:50]}...'"
                )

                intent_result = await self.intent_classifier_agent.classify(
                    query,
                    session_id=session_id,
                    user_id=customer_name or customer_id
                )
                intent = intent_result.intent
                category = intent_result.category
                confidence = intent_result.confidence

                logger.info(
                    f"[Intent Classification] Detected: {intent.value} "
                    f"(confidence: {confidence:.2f}) - {intent_result.reasoning}"
                )

                # ============================================================
                # STEP 1: Customer Lookup
                # ============================================================
                # Resolve customer_id if only name provided
                if not customer_id and customer_name:
                    customer_id = self.customer_repo.get_customer_id_by_name(
                        customer_name
                    )
                    if not customer_id:
                        raise NotFoundException("Customer", customer_name)

                # ============================================================
                # STEP 2: Route based on intent
                # ============================================================
                if intent == QueryIntent.INFORMATIONAL:
                    # Route to query answering service
                    logger.info(
                        f"[Informational Query] Routing to QueryAnsweringService "
                        f"for category: {category}"
                    )

                    result = await self._handle_informational_query(
                        query=query,
                        customer_id=customer_id,
                        customer_name=customer_name,
                        category=category,
                        extracted_info=intent_result.extracted_info,
                        start_time=start_time,
                        session_id=session_id,
                    )

                    # Update trace
                    if main_trace:
                        main_trace.update(output={
                            "intent": "informational",
                            "category": category.value if category else None,
                            "answer_length": len(result.reasoning),
                        })

                    return result

                else:
                    # Route to recommendation workflow
                    logger.info(
                        "[Recommendation Query] Routing to PersonalizedRecommendationWorkflow"
                    )

                    # Create agent context
                    context = AgentContext(
                        request_id=session_id,
                        session_id=session_id,
                        user_id=customer_id,
                        metadata={
                            "customer_name": customer_name,
                            "query": query,
                        },
                    )

                    # Execute workflow
                    result = await self.recommendation_workflow.execute(
                        customer_id=customer_id,
                        query=query,
                        top_n=top_n,
                        include_reasoning=include_reasoning,
                        context=context,
                    )

                    # Update trace
                    if main_trace:
                        main_trace.update(output={
                            "intent": "recommendation",
                            "recommendations_count": len(result.recommendations),
                            "processing_time_ms": result.processing_time_ms,
                            "confidence_score": result.confidence_score,
                        })

                    return result

            except Exception as e:
                # Log error and update trace
                logger.error(f"Recommendation workflow failed: {e}", exc_info=True)

                if main_trace:
                    main_trace.update(output={
                        "error": str(e),
                        "success": False,
                    })

                raise

    async def _handle_informational_query(
        self,
        query: str,
        customer_id: str,
        customer_name: Optional[str],
        category,
        extracted_info,
        start_time: float,
        session_id: str,
    ) -> RecommendationResponse:
        """
        Handle informational queries (e.g., "How much have I spent?")

        Args:
            query: User's query
            customer_id: Customer ID
            customer_name: Customer name (optional)
            category: Information category from intent classifier
            extracted_info: Extracted information dict
            start_time: Request start time
            session_id: Tracing session ID

        Returns:
            RecommendationResponse with answer in reasoning field (no products)
        """
        # Get customer profile
        profile = await self.customer_service.get_customer_profile(customer_id)

        # Generate answer using query answering service
        answer = await self.query_answering_service.answer_query(
            query=query,
            profile=profile,
            category=category,
            extracted_info=extracted_info,
            session_id=session_id,
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Return response with answer (no recommendations)
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
            recommendations=[],  # No products for informational queries
            reasoning=answer,
            confidence_score=0.9,  # High confidence for factual answers
            processing_time_ms=processing_time_ms,
            similar_customers_analyzed=0,
            products_considered=0,
            products_filtered_by_sentiment=0,
            recommendation_strategy="informational_query",
            agent_execution_order=["intent_classification", "query_answering"],
            metadata={
                "intent": "informational",
                "category": category.value if category else None,
                "session_id": session_id,
            },
        )
