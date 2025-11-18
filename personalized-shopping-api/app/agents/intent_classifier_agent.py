"""
Intent Classification Agent using LLM
Uses LangGraph to classify query intent and determine routing
"""

import json
import logging
from typing import Dict, Any, Literal, TypedDict, Annotated, Optional
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.models.llm_factory import get_llm, LLMType
from app.core.tracing import trace_span, log_event

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents"""
    INFORMATIONAL = "informational"  # Questions about customer data
    RECOMMENDATION = "recommendation"  # Product recommendation requests


class InformationCategory(str, Enum):
    """Categories of informational queries"""
    TOTAL_PURCHASES = "total_purchases"
    SPENDING = "spending"
    FAVORITE_CATEGORIES = "favorite_categories"
    RECENT_PURCHASES = "recent_purchases"
    CUSTOMER_PROFILE = "customer_profile"
    GENERAL = "general"


class IntentClassificationResult(BaseModel):
    """Result of intent classification"""
    intent: QueryIntent = Field(description="The classified intent")
    category: InformationCategory | None = Field(
        default=None,
        description="Category of informational query (if applicable)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the classification"
    )
    reasoning: str = Field(description="Explanation of the classification")
    extracted_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters extracted from query"
    )


class AgentState(TypedDict):
    """State for the intent classification agent"""
    query: str
    classification_result: Dict[str, Any] | None
    error: str | None
    session_id: Optional[str]
    user_id: Optional[str]


class IntentClassifierAgent:
    """
    LLM-based agent for classifying query intent and routing
    Uses LangGraph for orchestration
    """

    def __init__(self):
        self.llm = None
        self.graph = self._build_graph()

    def _get_llm(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Get LLM with tracing support

        Args:
            session_id: Session ID for tracing
            user_id: User ID for tracing
            metadata: Additional metadata for tracing
        """
        return get_llm(
            LLMType.RESPONSE,
            session_id=session_id,
            user_id=user_id,
            metadata={
                "agent": "intent_classifier",
                **(metadata or {})
            },
            tags=["intent_classification", "agent"]
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("extract_parameters", self._extract_parameters_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "extract_parameters")
        workflow.add_edge("extract_parameters", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """
        Node to classify the intent using LLM with tracing
        """
        query = state["query"]
        session_id = state.get("session_id")
        user_id = state.get("user_id")

        logger.info(f"[Intent Agent] Classifying query: '{query[:50]}...'")

        system_prompt = """You are an intelligent query classifier for a personalized shopping assistant API.

Your task is to analyze user queries and classify them into two main categories:

1. **INFORMATIONAL** - Questions asking about customer data:
   - Total purchases (e.g., "how many items has X bought?")
   - Spending (e.g., "how much has X spent?", "what's the total amount?")
   - Favorite categories (e.g., "what categories does X like?")
   - Recent purchases (e.g., "what did X buy recently?")
   - Customer profile (e.g., "tell me about X", "show X's profile")
   - General questions about customer data

2. **RECOMMENDATION** - Requests for product suggestions:
   - Direct requests for recommendations
   - Questions about what to buy
   - Requests for product suggestions
   - Questions about what customer would like

Analyze the query and respond with a JSON object with these fields:
- intent: "informational" or "recommendation"
- category: if informational, one of: "total_purchases", "spending", "favorite_categories", "recent_purchases", "customer_profile", "general"
- confidence: a number between 0 and 1
- reasoning: brief explanation of your classification

Examples:
Query: "what is the total purchase of Kenneth Martinez?"
Response: {"intent": "informational", "category": "total_purchases", "confidence": 0.95, "reasoning": "Query explicitly asks about total purchases"}

Query: "recommend some products for Kenneth Martinez"
Response: {"intent": "recommendation", "category": null, "confidence": 0.98, "reasoning": "Direct request for product recommendations"}

Query: "how much money has John spent?"
Response: {"intent": "informational", "category": "spending", "confidence": 0.92, "reasoning": "Query asks about spending amount"}

Now classify the following query. Respond ONLY with valid JSON, no additional text."""

        user_prompt = f"Query: {query}"

        try:
            llm = self._get_llm(
                session_id=session_id,
                user_id=user_id,
                metadata={"query": query[:100]}
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = llm.invoke(messages)
            response_text = response.content.strip()

            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
            else:
                result = json.loads(response_text)

            logger.info(
                f"[Intent Agent] Classified as: {result['intent']} "
                f"(confidence: {result['confidence']:.2f})"
            )

            # Log classification event to LangFuse
            log_event(
                name="intent_classified",
                metadata={
                    "query": query,
                    "intent": result['intent'],
                    "category": result.get('category'),
                    "confidence": result['confidence'],
                    "reasoning": result.get('reasoning', '')
                },
                level="DEFAULT"
            )

            state["classification_result"] = result
            state["error"] = None

        except Exception as e:
            logger.error(f"[Intent Agent] Classification failed: {e}")

            # Log error event to LangFuse
            log_event(
                name="intent_classification_error",
                metadata={
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                level="ERROR"
            )

            # Fallback to recommendation if classification fails
            state["classification_result"] = {
                "intent": "recommendation",
                "category": None,
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {str(e)}"
            }
            state["error"] = str(e)

        return state

    def _extract_parameters_node(self, state: AgentState) -> AgentState:
        """
        Node to extract additional parameters from the query
        """
        query = state["query"].lower()
        result = state["classification_result"]

        if not result:
            return state

        extracted_info = {}

        # Extract time periods
        if any(word in query for word in ["recent", "latest", "last"]):
            extracted_info["time_period"] = "recent"
        elif "all time" in query or "total" in query:
            extracted_info["time_period"] = "all_time"

        # Extract limits/top N
        import re
        top_match = re.search(r'\b(?:top|first|last)\s+(\d+)\b', query)
        if top_match:
            extracted_info["limit"] = int(top_match.group(1))

        result["extracted_info"] = extracted_info
        state["classification_result"] = result

        logger.info(f"[Intent Agent] Extracted parameters: {extracted_info}")

        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """
        Final node to validate and finalize the result
        """
        result = state["classification_result"]

        if result:
            # Ensure category is None for recommendation queries
            if result["intent"] == "recommendation":
                result["category"] = None

            # Ensure confidence is within bounds
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        logger.info("[Intent Agent] Classification complete")

        return state

    def classify(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> IntentClassificationResult:
        """
        Classify a query using the LLM-based agent

        Args:
            query: The user's query to classify
            session_id: Session ID for tracing
            user_id: User ID for tracing

        Returns:
            IntentClassificationResult with classification details
        """
        with trace_span(
            name="intent_classification",
            session_id=session_id,
            user_id=user_id,
            metadata={"query": query},
            tags=["agent", "intent_classification"],
            input_data={"query": query}
        ) as trace:
            initial_state = AgentState(
                query=query,
                classification_result=None,
                error=None,
                session_id=session_id,
                user_id=user_id
            )

            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Convert to result object
            result_dict = final_state["classification_result"]

            try:
                result = IntentClassificationResult(
                    intent=QueryIntent(result_dict["intent"]),
                    category=InformationCategory(result_dict["category"]) if result_dict.get("category") else None,
                    confidence=result_dict["confidence"],
                    reasoning=result_dict.get("reasoning", ""),
                    extracted_info=result_dict.get("extracted_info", {})
                )

                # Update trace with output
                if trace:
                    trace.update(output={
                        "intent": result.intent.value,
                        "category": result.category.value if result.category else None,
                        "confidence": result.confidence
                    })

                return result

            except Exception as e:
                logger.error(f"[Intent Agent] Failed to parse result: {e}")

                # Update trace with error
                if trace:
                    trace.update(
                        level="ERROR",
                        output={"error": str(e), "error_type": type(e).__name__}
                    )

                # Return safe fallback
                return IntentClassificationResult(
                    intent=QueryIntent.RECOMMENDATION,
                    category=None,
                    confidence=0.5,
                    reasoning=f"Fallback due to parsing error: {str(e)}",
                    extracted_info={}
                )

    def classify_dict(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify a query and return as dictionary (for backward compatibility)

        Args:
            query: The user's query to classify
            session_id: Session ID for tracing
            user_id: User ID for tracing

        Returns:
            Dictionary with classification details
        """
        result = self.classify(query, session_id=session_id, user_id=user_id)
        return {
            "intent": result.intent,
            "category": result.category,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "extracted_info": result.extracted_info,
        }
