"""
Main CLI entry point for Personalized Shopping Assistant
"""

import sys
import argparse
import logging
from pathlib import Path

from src.config import config
from src.utils.logging import setup_logging
from src.utils.validators import validate_input
from src.graph.workflow import run_workflow
from src.memory.conversation_store import ConversationStore
from src.utils.metrics import get_metrics_collector

def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🛍️  Personalized Shopping Assistant")
    print("="*60 + "\n")

def print_response(state: dict):
    """
    Format and print the response

    Args:
        state: Final workflow state
    """
    print("\n" + "━"*60)

    # Print errors if any
    errors = state.get("errors", [])
    if errors:
        print("❌ Errors:")
        for error in errors:
            print(f"   - {error}")
        print()

    # Print warnings if any
    warnings = state.get("warnings", [])
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()

    # Print response
    response = state.get("final_response", "No response generated")
    print(response)
    print("\n" + "━"*60)

    # Print metadata
    print("\n📊 Metadata:")
    print(f"   - Processing time: {state.get('processing_time_ms', 0):.2f}ms")
    print(f"   - Confidence: {state.get('confidence_score', 0):.2%}")

    agent_order = state.get("agent_execution_order", [])
    if agent_order:
        print(f"   - Agents executed: {' → '.join(agent_order)}")

    # Print recommendations summary
    recommendations = state.get("final_recommendations", [])
    if recommendations:
        print(f"   - Recommendations: {len(recommendations)}")

    similar_customers = state.get("similar_customers", [])
    if similar_customers:
        print(f"   - Similar customers: {len(similar_customers)}")

    filtered_products = state.get("filtered_products", [])
    products_filtered_out = state.get("products_filtered_out", 0)
    if filtered_products:
        print(f"   - Products evaluated: {len(filtered_products) + products_filtered_out}")
        print(f"   - Products filtered out: {products_filtered_out}")

    print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Personalized Shopping Assistant - Get recommendations based on customer behavior"
    )

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Natural language query (e.g., 'What would John Smith like?')"
    )

    parser.add_argument(
        "--customer-name",
        type=str,
        help="Customer name"
    )

    parser.add_argument(
        "--customer-id",
        type=str,
        help="Customer ID"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save conversation to history"
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Show detailed performance metrics"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    # Interactive mode if no query provided
    if not args.query:
        print_banner()
        print("Interactive Mode")
        print("Type your query or 'exit' to quit\n")

        while True:
            try:
                query = input("Query: ").strip()

                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                # Ask for customer name if not provided
                customer_name = args.customer_name
                if not customer_name:
                    customer_name = input("Customer name (or ID): ").strip()

                # Determine if it's an ID or name
                customer_id = None
                if customer_name.isdigit():
                    customer_id = customer_name
                    customer_name = None

                # Run workflow
                state = run_workflow(
                    query=query,
                    customer_name=customer_name,
                    customer_id=customer_id
                )

                # Print response
                print_response(state)

                # Save conversation
                if not args.no_save:
                    conv_store = ConversationStore()
                    conv_store.save_conversation(
                        session_id=state["session_id"],
                        query=query,
                        response=state.get("final_response", ""),
                        metadata={
                            "customer_name": customer_name,
                            "customer_id": customer_id,
                            "processing_time_ms": state.get("processing_time_ms", 0)
                        }
                    )

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\n❌ Error: {e}\n")

        return

    # Single query mode
    query = args.query

    # Validate input
    validation = validate_input(
        query=query,
        customer_name=args.customer_name,
        customer_id=args.customer_id
    )

    if not validation['valid']:
        print("❌ Invalid input:")
        for error in validation['errors']:
            print(f"   - {error}")
        sys.exit(1)

    print_banner()

    # Run workflow
    try:
        state = run_workflow(
            query=query,
            customer_name=args.customer_name,
            customer_id=args.customer_id
        )

        # Print response
        print_response(state)

        # Save conversation
        if not args.no_save:
            conv_store = ConversationStore()
            conv_store.save_conversation(
                session_id=state["session_id"],
                query=query,
                response=state.get("final_response", ""),
                metadata={
                    "customer_name": args.customer_name,
                    "customer_id": args.customer_id,
                    "processing_time_ms": state.get("processing_time_ms", 0)
                }
            )

        # Show metrics if requested
        if args.metrics:
            metrics = get_metrics_collector()
            summary = metrics.get_summary()

            print("\n📈 Performance Metrics:")
            print("="*60)

            for metric_name, stats in summary['metrics'].items():
                print(f"\n{metric_name}:")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
                print(f"  Avg: {stats['avg']:.2f}")
                print(f"  Count: {stats['count']}")

            if summary['counters']:
                print("\n\nCounters:")
                for counter_name, value in summary['counters'].items():
                    print(f"  {counter_name}: {value}")

            print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
