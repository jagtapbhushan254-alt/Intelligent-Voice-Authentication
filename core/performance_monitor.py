"""
SecureX-Assist - Performance Monitoring
Real-time performance metrics and optimization tracking
"""

import logging
import time
from typing import Dict, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for voice assistant operations"""
    transcription_times: deque = field(default_factory=lambda: deque(maxlen=100))
    classification_times: deque = field(default_factory=lambda: deque(maxlen=100))
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    tts_times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_transcription(self, duration: float):
        self.transcription_times.append(duration)
    
    def add_classification(self, duration: float):
        self.classification_times.append(duration)
    
    def add_execution(self, duration: float):
        self.execution_times.append(duration)
    
    def add_tts(self, duration: float):
        self.tts_times.append(duration)
    
    def add_total(self, duration: float):
        self.total_times.append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        def calc_stats(times: deque) -> Dict[str, float]:
            if not times:
                return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            return {
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        
        return {
            'transcription': calc_stats(self.transcription_times),
            'classification': calc_stats(self.classification_times),
            'execution': calc_stats(self.execution_times),
            'tts': calc_stats(self.tts_times),
            'total': calc_stats(self.total_times)
        }
    
    def get_summary(self) -> str:
        """Get human-readable performance summary"""
        stats = self.get_stats()
        total = stats['total']
        
        if total['count'] == 0:
            return "No performance data available"
        
        return (
            f"Performance Summary (last {total['count']} commands):\n"
            f"  Total Response Time: {total['avg']:.2f}s avg ({total['min']:.2f}s - {total['max']:.2f}s)\n"
            f"  Transcription: {stats['transcription']['avg']:.2f}s avg\n"
            f"  Classification: {stats['classification']['avg']:.2f}s avg\n"
            f"  Execution: {stats['execution']['avg']:.2f}s avg\n"
            f"  TTS: {stats['tts']['avg']:.2f}s avg"
        )


# Global metrics instance
_performance_metrics = None


def get_performance_metrics() -> PerformanceMetrics:
    """Get or create the global performance metrics instance"""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetrics()
    return _performance_metrics


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        logger.debug(f"⏱️ {self.operation_name} took {self.duration:.3f}s")
        return False
