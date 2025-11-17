"""
LangGraph workflow definition for personalized shopping assistant
"""

import time
from typing import Dict
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state import ShoppingAssistantState, initialize_state
from ..agents import (
    customer_profiling_agent,
    similar_customers_agent,
    review_filtering_agent,
    recommendation_agent,
    response_generation_agent
)
from .routers import (
    route_after_profiling,
    route_after_similar_customers,
    route_after_filtering,
    route_after_recommendation
)

logger = logging.getLogger(__name__)

# ==================== Workflow Node Wrappers ====================

def profiling_node(state: ShoppingAssistantState) -> Dict:
    """Wrapper for customer profiling agent"""
    logger.info("\n" + "="*60)
    logger.info("[Agent 1: Customer Profiling]")
    logger.info("="*60)
    return customer_profiling_agent(state)

def similar_customers_node(state: ShoppingAssistantState) -> Dict:
    """Wrapper for similar customers agent"""
    logger.info("\n" + "="*60)
    logger.info("[Agent 2: Similar Customer Discovery]")
    logger.info("="*60)
    return similar_customers_agent(state)

def review_filtering_node(state: ShoppingAssistantState) -> Dict:
    """Wrapper for review filtering agent"""
    logger.info("\n" + "="*60)
    logger.info("[Agent 3: Review-Based Filtering]")
    logger.info("="*60)
    return review_filtering_agent(state)

def recommendation_node(state: ShoppingAssistantState) -> Dict:
    """Wrapper for recommendation agent"""
    logger.info("\n" + "="*60)
    logger.info("[Agent 4: Cross-Category Recommendation]")
    logger.info("="*60)
    return recommendation_agent(state)

def response_node(state: ShoppingAssistantState) -> Dict:
    """Wrapper for response generation agent"""
    logger.info("\n" + "="*60)
    logger.info("[Agent 5: Response Generation]")
    logger.info("="*60)
    return response_generation_agent(state)

def error_handler_node(state: ShoppingAssistantState) -> Dict:
    """Handle errors and generate error response"""
    logger.error("Error handler invoked")
    errors = state.get("errors", [])

    error_response = "I apologize, but I encountered an issue:\n"
    if errors:
        error_response += f"- {errors[0]}"
    else:
        error_response += "- Unable to process your request"

    return {
        "final_response": error_response,
        "agent_execution_order": ["error_handler"]
    }

def fallback_recommendation_node(state: ShoppingAssistantState) -> Dict:
    """Provide fallback recommendations when similar customers not found"""
    logger.warning("Using fallback recommendations")

    # Simple fallback: return most popular products
    fallback_response = """I don't have enough data about similar customers to make personalized recommendations.
However, I can suggest exploring our most popular products in the categories you're interested in.
Please try again later as we gather more customer data."""

    return {
        "final_response": fallback_response,
        "fallback_used": True,
        "agent_execution_order": ["fallback_recommendation"]
    }

# ==================== Workflow Creation ====================

def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow

    Workflow Structure:
    START → profiling → [profile_found?]
        → similar_customers → [found_similar?]
        → review_filtering → [enough_products?]
        → recommendation → response_generation → END

    Returns:
        Compiled StateGraph
    """
    logger.info("Creating workflow graph...")

    # Initialize graph
    workflow = StateGraph(ShoppingAssistantState)

    # Add nodes
    workflow.add_node("profiling", profiling_node)
    workflow.add_node("similar_customers", similar_customers_node)
    workflow.add_node("review_filtering", review_filtering_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("response_generation", response_node)
    workflow.add_node("error", error_handler_node)
    workflow.add_node("fallback", fallback_recommendation_node)

    # Set entry point
    workflow.set_entry_point("profiling")

    # Add conditional edges
    workflow.add_conditional_edges(
        "profiling",
        route_after_profiling,
        {
            "similar_customers": "similar_customers",
            "error": "error"
        }
    )

    workflow.add_conditional_edges(
        "similar_customers",
        route_after_similar_customers,
        {
            "review_filtering": "review_filtering",
            "fallback": "fallback"
        }
    )

    workflow.add_conditional_edges(
        "review_filtering",
        route_after_filtering,
        {
            "recommendation": "recommendation",
            "popular_fallback": "fallback"
        }
    )

    # Direct edges
    workflow.add_edge("recommendation", "response_generation")
    workflow.add_edge("response_generation", END)
    workflow.add_edge("error", END)
    workflow.add_edge("fallback", END)

    # Compile with checkpointing
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)

    logger.info("Workflow created successfully")
    return compiled_workflow

# ==================== Workflow Execution ====================

def run_workflow(
    query: str,
    customer_name: str = None,
    customer_id: str = None
) -> Dict:
    """
    Run the complete workflow

    Args:
        query: User query
        customer_name: Customer name
        customer_id: Customer ID

    Returns:
        Final state dictionary
    """
    start_time = time.time()

    logger.info("\n" + "🔍" + "="*59)
    logger.info(f"Processing Query: {query}")
    logger.info("="*60 + "\n")

    # Initialize state
    initial_state = initialize_state(
        query=query,
        customer_name=customer_name,
        customer_id=customer_id
    )

    # Create workflow
    workflow = create_workflow()

    # Execute workflow
    try:
        # Run with config for checkpointing
        config = {"configurable": {"thread_id": initial_state["session_id"]}}

        final_state = None
        for state in workflow.stream(initial_state, config):
            # Stream yields state updates
            final_state = state

        # Get the last state (after END)
        if final_state:
            # Extract the actual state (LangGraph wraps it)
            if isinstance(final_state, dict):
                # Get the last node's output
                final_state = list(final_state.values())[-1]

        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000

        if final_state:
            final_state["processing_time_ms"] = processing_time_ms

        logger.info("\n" + "="*60)
        logger.info(f"✓ Workflow completed in {processing_time_ms:.2f}ms")
        logger.info("="*60 + "\n")

        return final_state

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)

        # Return error state
        return {
            **initial_state,
            "errors": [f"Workflow error: {str(e)}"],
            "final_response": f"I apologize, but an error occurred: {str(e)}",
            "processing_time_ms": (time.time() - start_time) * 1000
        }
