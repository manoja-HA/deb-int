"""
Example: Using Prompt Version Tracking and A/B Testing

This file shows how to:
1. Set up A/B test experiments for prompts
2. Track metrics for different prompt versions
3. Automatically rollback on errors
4. Compare performance across variants
"""

import asyncio
import logging
from typing import Optional

from app.prompts.version_tracker import (
    get_version_tracker,
    PromptVariant
)
from app.prompts.loader import get_prompt_loader

logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Setting up an A/B test
# ============================================================================

def setup_response_generation_ab_test():
    """
    Example: Test two different versions of response generation prompt

    Scenario:
    - v1.0.0: Current production prompt (conservative, factual)
    - v2.0.0: New experimental prompt (more engaging, personalized)

    We want to test which version gets better user engagement.
    """
    tracker = get_version_tracker()

    # Create experiment
    experiment = tracker.create_experiment(
        experiment_id="response_gen_engagement_test",
        prompt_id="response.generation",
        variants=[
            PromptVariant(
                variant_id="v1.0.0",
                prompt_id="response.generation",
                traffic_percentage=50.0,  # 50% of traffic
                enabled=True,
                metadata={
                    "description": "Conservative, factual prompt",
                    "version": "1.0.0"
                }
            ),
            PromptVariant(
                variant_id="v2.0.0",
                prompt_id="response.generation",
                traffic_percentage=50.0,  # 50% of traffic
                enabled=True,
                metadata={
                    "description": "Engaging, personalized prompt",
                    "version": "2.0.0"
                }
            ),
        ],
        success_metric="user_engagement_score",  # Custom metric
        metadata={
            "hypothesis": "More engaging prompts increase user satisfaction",
            "owner": "ml_team",
            "start_date": "2024-01-15"
        }
    )

    logger.info(f"Created A/B test: {experiment.experiment_id}")
    return experiment


# ============================================================================
# Example 2: Using variant in agent
# ============================================================================

async def generate_response_with_ab_test(
    query: str,
    customer_name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Example: Use A/B tested prompt variant in response generation

    The version tracker automatically assigns a variant based on:
    - Active experiment configuration
    - Traffic percentages
    - Sticky sessions (same user always gets same variant)
    """
    tracker = get_version_tracker()
    prompt_loader = get_prompt_loader()

    # Get variant for this request
    variant_id = tracker.get_variant(
        prompt_id="response.generation",
        user_id=user_id,
        session_id=session_id
    )

    # If experiment active, use variant; otherwise use default
    if variant_id:
        logger.info(f"Using variant '{variant_id}' for response generation")
        # Load specific version from prompts/response_generation/v2.0.0/
        # (You would organize prompts by version in subdirectories)
    else:
        logger.info("No active experiment, using default prompt")
        variant_id = "default"

    # Track invocation start time
    import time
    start_time = time.time()

    try:
        # Generate response (simplified example)
        # In real code, you'd call ResponseGenerationAgentV2 here
        response = f"Response generated for {customer_name} using variant {variant_id}"
        success = True
        confidence = 0.85  # From agent output

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        success = False
        confidence = 0.0
        response = None

    # Track metrics
    latency_ms = (time.time() - start_time) * 1000
    tracker.track_invocation(
        prompt_id="response.generation",
        variant_id=variant_id,
        success=success,
        latency_ms=latency_ms,
        confidence=confidence,
        custom_metrics={
            "user_engagement_score": 0.78,  # Would come from user feedback
        }
    )

    # Check if we need to rollback any variant
    tracker.check_auto_rollback(
        prompt_id="response.generation",
        error_rate_threshold=0.10,  # Rollback if >10% errors
        min_invocations=50  # Only after 50+ requests
    )

    return response


# ============================================================================
# Example 3: Analyzing experiment results
# ============================================================================

def analyze_experiment_results(prompt_id: str):
    """
    Example: Analyze A/B test results

    Compare metrics across variants to determine winner.
    """
    tracker = get_version_tracker()

    # Get all variant metrics
    metrics_map = tracker.get_experiment_metrics(prompt_id)

    if not metrics_map:
        logger.warning(f"No metrics found for prompt '{prompt_id}'")
        return

    print(f"\n{'='*70}")
    print(f"A/B Test Results: {prompt_id}")
    print(f"{'='*70}\n")

    for variant_id, metrics in metrics_map.items():
        if not metrics:
            continue

        print(f"Variant: {variant_id}")
        print(f"  Total Invocations:   {metrics.total_invocations}")
        print(f"  Success Rate:        {(1 - metrics.error_rate) * 100:.1f}%")
        print(f"  Avg Latency:         {metrics.avg_latency_ms:.1f}ms")
        print(f"  P95 Latency:         {metrics.p95_latency_ms:.1f}ms")
        print(f"  Avg Confidence:      {metrics.avg_confidence:.2f}")

        if "user_engagement_score" in metrics.custom_metrics:
            print(f"  User Engagement:     {metrics.custom_metrics['user_engagement_score']:.2f}")

        print()

    # Determine winner (simplified - in real scenario use statistical significance)
    best_variant = max(
        metrics_map.items(),
        key=lambda x: x[1].avg_confidence if x[1] else 0
    )

    print(f"🏆 Winner: {best_variant[0]} (Avg Confidence: {best_variant[1].avg_confidence:.2f})")
    print(f"{'='*70}\n")


# ============================================================================
# Example 4: Gradual rollout (canary deployment)
# ============================================================================

def setup_canary_deployment():
    """
    Example: Gradual rollout of new prompt version

    Start with 5% traffic, then gradually increase if metrics look good.
    """
    tracker = get_version_tracker()

    # Initial deployment: 5% on new version
    experiment = tracker.create_experiment(
        experiment_id="intent_classifier_v2_canary",
        prompt_id="intent.classification",
        variants=[
            PromptVariant(
                variant_id="v1.0.0",
                prompt_id="intent.classification",
                traffic_percentage=95.0,  # 95% on stable
                enabled=True
            ),
            PromptVariant(
                variant_id="v2.0.0",
                prompt_id="intent.classification",
                traffic_percentage=5.0,  # 5% on new version
                enabled=True,
                metadata={"canary": True}
            ),
        ],
        success_metric="classification_accuracy"
    )

    logger.info("Canary deployment started: 5% traffic on v2.0.0")

    # After monitoring, you can manually adjust traffic percentages:
    # experiment.variants[0].traffic_percentage = 80  # v1.0.0
    # experiment.variants[1].traffic_percentage = 20  # v2.0.0
    # ... and eventually roll out to 100%

    return experiment


# ============================================================================
# Example 5: Automatic rollback on high error rate
# ============================================================================

async def demonstrate_auto_rollback():
    """
    Example: Automatic rollback if variant has high error rate

    If a new prompt version causes errors, it's automatically disabled.
    """
    tracker = get_version_tracker()

    # Simulate 100 requests with v2.0.0 having 15% error rate
    for i in range(100):
        variant_id = "v2.0.0" if i % 2 == 0 else "v1.0.0"

        # v2.0.0 has 15% error rate
        success = True
        if variant_id == "v2.0.0" and i % 7 == 0:  # ~15% error rate
            success = False

        tracker.track_invocation(
            prompt_id="response.generation",
            variant_id=variant_id,
            success=success,
            latency_ms=150.0,
            confidence=0.85 if success else 0.0
        )

    # Check for rollback (triggers if error rate > 10%)
    tracker.check_auto_rollback(
        prompt_id="response.generation",
        error_rate_threshold=0.10,
        min_invocations=50
    )

    # Check if v2.0.0 was paused
    experiment = tracker._experiments.get("response.generation")
    if experiment:
        for variant in experiment.variants:
            if variant.variant_id == "v2.0.0":
                if not variant.enabled:
                    logger.warning("✅ Auto-rollback triggered! v2.0.0 was paused due to high error rate")
                else:
                    logger.info("v2.0.0 still enabled (error rate below threshold)")


# ============================================================================
# Main Example
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Prompt Version Tracking & A/B Testing Examples")
    print("="*70 + "\n")

    # Example 1: Setup A/B test
    print("1. Setting up A/B test...")
    setup_response_generation_ab_test()

    # Example 2: Generate responses with A/B test
    print("\n2. Generating responses with A/B test...")
    for i in range(10):
        await generate_response_with_ab_test(
            query="What should I buy?",
            customer_name="Kenneth Martinez",
            user_id=f"user_{i % 3}",  # 3 users, sticky sessions
            session_id=f"session_{i}"
        )

    # Example 3: Analyze results
    print("\n3. Analyzing experiment results...")
    analyze_experiment_results("response.generation")

    # Example 4: Canary deployment
    print("\n4. Setting up canary deployment...")
    setup_canary_deployment()

    # Example 5: Auto rollback
    print("\n5. Demonstrating automatic rollback...")
    await demonstrate_auto_rollback()

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
