"""
Performance tracking and metrics collection
"""

import time
import logging
from typing import Dict, Callable, Any
from functools import wraps
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and track performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)

    def record_latency(self, agent_name: str, latency_ms: float) -> None:
        """
        Record agent execution latency

        Args:
            agent_name: Name of the agent
            latency_ms: Execution time in milliseconds
        """
        self.metrics[f"{agent_name}_latency_ms"].append(latency_ms)
        logger.debug(f"{agent_name} latency: {latency_ms:.2f}ms")

    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """
        Increment a counter metric

        Args:
            counter_name: Name of the counter
            value: Amount to increment
        """
        self.counters[counter_name] += value

    def record_metric(self, metric_name: str, value: Any) -> None:
        """
        Record a custom metric

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)

    def get_stats(self, metric_name: str) -> Dict:
        """
        Get statistics for a metric

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with min, max, avg, count
        """
        values = self.metrics.get(metric_name, [])

        if not values:
            return {
                'min': 0,
                'max': 0,
                'avg': 0,
                'count': 0
            }

        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }

    def get_summary(self) -> Dict:
        """
        Get summary of all metrics

        Returns:
            Dictionary with all metrics and counters
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'counters': dict(self.counters),
            'metrics': {}
        }

        for metric_name in self.metrics:
            summary['metrics'][metric_name] = self.get_stats(metric_name)

        return summary

    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()

# Global metrics collector
_metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return _metrics_collector

def track_agent_performance(agent_name: str) -> Callable:
    """
    Decorator to track agent performance

    Args:
        agent_name: Name of the agent

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record success
                _metrics_collector.increment_counter(f"{agent_name}_success")

                return result

            except Exception as e:
                # Record failure
                _metrics_collector.increment_counter(f"{agent_name}_failure")
                logger.error(f"{agent_name} failed: {e}")
                raise

            finally:
                # Record latency
                elapsed_ms = (time.time() - start_time) * 1000
                _metrics_collector.record_latency(agent_name, elapsed_ms)

        return wrapper
    return decorator

class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name} took {self.elapsed_ms:.2f}ms")
