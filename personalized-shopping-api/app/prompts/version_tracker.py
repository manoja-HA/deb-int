"""
Prompt Version Tracking and A/B Testing

This module provides infrastructure for:
- Tracking prompt versions and performance metrics
- A/B testing different prompts
- Automatic rollback on performance degradation
- Experiment management
"""

import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from collections import defaultdict
import random

from app.core.tracing import log_event

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class PromptVariant(BaseModel):
    """A prompt variant for A/B testing"""
    variant_id: str = Field(description="Unique variant identifier (e.g., 'v1.0.0', 'experiment-v2')")
    prompt_id: str = Field(description="Base prompt ID (e.g., 'response.generation')")
    traffic_percentage: float = Field(
        ge=0,
        le=100,
        default=0,
        description="Percentage of traffic to route to this variant"
    )
    enabled: bool = Field(default=True, description="Whether this variant is active")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class PromptExperiment(BaseModel):
    """A/B test experiment configuration"""
    experiment_id: str = Field(description="Unique experiment identifier")
    prompt_id: str = Field(description="Prompt being tested")
    variants: List[PromptVariant] = Field(description="Variants in the experiment")
    start_date: datetime = Field(default_factory=datetime.now)
    end_date: Optional[datetime] = Field(default=None)
    status: Literal["active", "paused", "completed"] = Field(default="active")
    success_metric: str = Field(
        default="confidence_score",
        description="Metric to optimize (e.g., 'confidence_score', 'user_rating')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptMetrics(BaseModel):
    """Performance metrics for a prompt variant"""
    variant_id: str
    prompt_id: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Version Tracker
# ============================================================================

class PromptVersionTracker:
    """
    Tracks prompt versions and manages A/B testing

    Features:
    - Traffic splitting for A/B tests
    - Performance metric tracking
    - Automatic rollback on errors
    - Experiment management

    Usage:
        tracker = PromptVersionTracker()

        # Create experiment
        tracker.create_experiment(
            experiment_id="response_gen_test_v2",
            prompt_id="response.generation",
            variants=[
                PromptVariant(variant_id="v1.0.0", traffic_percentage=50),
                PromptVariant(variant_id="v2.0.0", traffic_percentage=50),
            ]
        )

        # Get variant for request
        variant = tracker.get_variant("response.generation", user_id="user123")

        # Track metrics
        tracker.track_invocation(
            prompt_id="response.generation",
            variant_id="v2.0.0",
            success=True,
            latency_ms=145.2,
            confidence=0.89
        )
    """

    def __init__(self):
        # Experiments by prompt_id
        self._experiments: Dict[str, PromptExperiment] = {}

        # Metrics by variant_id
        self._metrics: Dict[str, PromptMetrics] = {}

        # Sticky sessions for consistent variant assignment
        self._user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Latency buckets for P95 calculation
        self._latency_buckets: Dict[str, List[float]] = defaultdict(list)

    def create_experiment(
        self,
        experiment_id: str,
        prompt_id: str,
        variants: List[PromptVariant],
        success_metric: str = "confidence_score",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptExperiment:
        """
        Create a new A/B test experiment

        Args:
            experiment_id: Unique experiment ID
            prompt_id: Prompt to test (e.g., "response.generation")
            variants: List of variants with traffic percentages
            success_metric: Metric to optimize
            metadata: Additional experiment metadata

        Returns:
            Created experiment
        """
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

        experiment = PromptExperiment(
            experiment_id=experiment_id,
            prompt_id=prompt_id,
            variants=variants,
            success_metric=success_metric,
            metadata=metadata or {}
        )

        self._experiments[prompt_id] = experiment

        # Initialize metrics for each variant
        for variant in variants:
            if variant.variant_id not in self._metrics:
                self._metrics[variant.variant_id] = PromptMetrics(
                    variant_id=variant.variant_id,
                    prompt_id=prompt_id
                )

        logger.info(
            f"[Prompt A/B Test] Created experiment '{experiment_id}' for '{prompt_id}' "
            f"with {len(variants)} variants"
        )

        # Log to LangFuse
        log_event(
            name="prompt_experiment_created",
            metadata={
                "experiment_id": experiment_id,
                "prompt_id": prompt_id,
                "variants": [v.variant_id for v in variants],
                "traffic_split": {v.variant_id: v.traffic_percentage for v in variants}
            }
        )

        return experiment

    def get_variant(
        self,
        prompt_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the variant to use for a request

        Uses sticky sessions to ensure consistent experience for the same user.
        If no experiment exists, returns None (use default prompt).

        Args:
            prompt_id: Prompt ID
            user_id: User ID for sticky assignment (optional)
            session_id: Session ID for sticky assignment (optional)

        Returns:
            Variant ID to use, or None for default
        """
        experiment = self._experiments.get(prompt_id)

        if not experiment or experiment.status != "active":
            return None

        # Check if experiment has ended
        if experiment.end_date and datetime.now() > experiment.end_date:
            self.stop_experiment(prompt_id)
            return None

        # Sticky session key
        sticky_key = user_id or session_id
        if sticky_key and sticky_key in self._user_assignments.get(prompt_id, {}):
            # Return previously assigned variant
            return self._user_assignments[prompt_id][sticky_key]

        # Assign new variant based on traffic percentages
        enabled_variants = [v for v in experiment.variants if v.enabled]
        if not enabled_variants:
            return None

        # Weighted random selection
        rand_value = random.uniform(0, 100)
        cumulative = 0.0

        for variant in enabled_variants:
            cumulative += variant.traffic_percentage
            if rand_value <= cumulative:
                variant_id = variant.variant_id

                # Save assignment for sticky sessions
                if sticky_key:
                    self._user_assignments[prompt_id][sticky_key] = variant_id

                logger.debug(
                    f"[Prompt A/B Test] Assigned variant '{variant_id}' for '{prompt_id}'"
                )

                return variant_id

        # Fallback (shouldn't happen if percentages sum to 100)
        return enabled_variants[-1].variant_id

    def track_invocation(
        self,
        prompt_id: str,
        variant_id: str,
        success: bool,
        latency_ms: float,
        confidence: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Track metrics for a prompt invocation

        Args:
            prompt_id: Prompt ID
            variant_id: Variant ID used
            success: Whether invocation succeeded
            latency_ms: Latency in milliseconds
            confidence: Confidence score (0-1)
            custom_metrics: Additional metrics to track
        """
        if variant_id not in self._metrics:
            self._metrics[variant_id] = PromptMetrics(
                variant_id=variant_id,
                prompt_id=prompt_id
            )

        metrics = self._metrics[variant_id]

        # Update counts
        metrics.total_invocations += 1
        if success:
            metrics.successful_invocations += 1
        else:
            metrics.failed_invocations += 1

        # Update latency (rolling average)
        prev_total = metrics.total_invocations - 1
        if prev_total > 0:
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * prev_total + latency_ms) / metrics.total_invocations
            )
        else:
            metrics.avg_latency_ms = latency_ms

        # Track latency for P95
        self._latency_buckets[variant_id].append(latency_ms)
        if len(self._latency_buckets[variant_id]) > 1000:
            # Keep only recent 1000 samples
            self._latency_buckets[variant_id] = self._latency_buckets[variant_id][-1000:]

        # Calculate P95
        if self._latency_buckets[variant_id]:
            sorted_latencies = sorted(self._latency_buckets[variant_id])
            p95_idx = int(len(sorted_latencies) * 0.95)
            metrics.p95_latency_ms = sorted_latencies[p95_idx]

        # Update confidence (rolling average)
        if confidence is not None:
            if prev_total > 0:
                metrics.avg_confidence = (
                    (metrics.avg_confidence * prev_total + confidence) / metrics.total_invocations
                )
            else:
                metrics.avg_confidence = confidence

        # Update error rate
        metrics.error_rate = metrics.failed_invocations / metrics.total_invocations

        # Update custom metrics
        if custom_metrics:
            for key, value in custom_metrics.items():
                if key not in metrics.custom_metrics:
                    metrics.custom_metrics[key] = value
                else:
                    # Rolling average
                    metrics.custom_metrics[key] = (
                        (metrics.custom_metrics[key] * prev_total + value) / metrics.total_invocations
                    )

        metrics.last_updated = datetime.now()

        # Log periodic metrics (every 100 invocations)
        if metrics.total_invocations % 100 == 0:
            logger.info(
                f"[Prompt Metrics] {variant_id}: "
                f"{metrics.total_invocations} invocations, "
                f"{metrics.error_rate*100:.1f}% error rate, "
                f"{metrics.avg_latency_ms:.0f}ms avg latency, "
                f"{metrics.avg_confidence:.2f} avg confidence"
            )

            # Log to LangFuse
            log_event(
                name="prompt_metrics_update",
                metadata={
                    "variant_id": variant_id,
                    "prompt_id": prompt_id,
                    "total_invocations": metrics.total_invocations,
                    "error_rate": metrics.error_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "avg_confidence": metrics.avg_confidence,
                    "p95_latency_ms": metrics.p95_latency_ms
                }
            )

    def get_metrics(self, variant_id: str) -> Optional[PromptMetrics]:
        """Get metrics for a variant"""
        return self._metrics.get(variant_id)

    def get_experiment_metrics(self, prompt_id: str) -> Dict[str, PromptMetrics]:
        """Get metrics for all variants in an experiment"""
        experiment = self._experiments.get(prompt_id)
        if not experiment:
            return {}

        return {
            v.variant_id: self._metrics.get(v.variant_id)
            for v in experiment.variants
            if v.variant_id in self._metrics
        }

    def stop_experiment(self, prompt_id: str):
        """Stop an experiment"""
        if prompt_id in self._experiments:
            self._experiments[prompt_id].status = "completed"
            self._experiments[prompt_id].end_date = datetime.now()

            logger.info(f"[Prompt A/B Test] Stopped experiment for '{prompt_id}'")

            log_event(
                name="prompt_experiment_stopped",
                metadata={
                    "experiment_id": self._experiments[prompt_id].experiment_id,
                    "prompt_id": prompt_id
                }
            )

    def pause_variant(self, prompt_id: str, variant_id: str):
        """
        Pause a variant (automatic rollback on errors)

        This can be used for automatic rollback if a variant shows
        high error rates or performance degradation.
        """
        experiment = self._experiments.get(prompt_id)
        if not experiment:
            return

        for variant in experiment.variants:
            if variant.variant_id == variant_id:
                variant.enabled = False

                logger.warning(
                    f"[Prompt A/B Test] Paused variant '{variant_id}' for '{prompt_id}'"
                )

                log_event(
                    name="prompt_variant_paused",
                    metadata={
                        "variant_id": variant_id,
                        "prompt_id": prompt_id,
                        "experiment_id": experiment.experiment_id
                    },
                    level="WARNING"
                )
                break

    def check_auto_rollback(
        self,
        prompt_id: str,
        error_rate_threshold: float = 0.10,
        min_invocations: int = 50
    ):
        """
        Check if any variant should be auto-rolled back

        Args:
            prompt_id: Prompt ID to check
            error_rate_threshold: Error rate threshold (default 10%)
            min_invocations: Minimum invocations before checking
        """
        experiment = self._experiments.get(prompt_id)
        if not experiment or experiment.status != "active":
            return

        for variant in experiment.variants:
            if not variant.enabled:
                continue

            metrics = self._metrics.get(variant.variant_id)
            if not metrics or metrics.total_invocations < min_invocations:
                continue

            # Check error rate
            if metrics.error_rate > error_rate_threshold:
                logger.error(
                    f"[Prompt A/B Test] Variant '{variant.variant_id}' has high error rate "
                    f"({metrics.error_rate*100:.1f}%), triggering rollback"
                )
                self.pause_variant(prompt_id, variant.variant_id)


# ============================================================================
# Global Tracker Instance
# ============================================================================

_tracker_instance: Optional[PromptVersionTracker] = None


def get_version_tracker() -> PromptVersionTracker:
    """Get global prompt version tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PromptVersionTracker()
    return _tracker_instance
