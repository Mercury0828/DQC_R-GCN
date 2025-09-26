"""
Performance monitoring utilities for DQC_R-GCN.
"""

import time
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

class PerformanceMonitor:
    """Global performance monitor for tracking execution times."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
        self.enabled = True
    
    def start_timer(self, name: str):
        """Start a named timer."""
        if self.enabled:
            self.current_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str):
        """End a named timer and record the duration."""
        if self.enabled and name in self.current_timers:
            duration = time.perf_counter() - self.current_timers[name]
            self.timings[name].append(duration)
            del self.current_timers[name]
            return duration
        return 0
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing a code block."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for a named timer."""
        if name not in self.timings or not self.timings[name]:
            return {}
        
        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times),
            'last': times[-1]
        }
    
    def print_summary(self, top_n: int = 10):
        """Print summary of all timings."""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Sort by total time
        sorted_timings = sorted(
            [(name, self.get_stats(name)) for name in self.timings],
            key=lambda x: x[1].get('total', 0) if x[1] else 0,
            reverse=True
        )
        
        for name, stats in sorted_timings[:top_n]:
            if stats:
                print(f"\n{name}:")
                print(f"  Calls: {stats['count']}")
                print(f"  Total: {stats['total']:.3f}s")
                print(f"  Mean:  {stats['mean']:.6f}s")
                print(f"  Std:   {stats['std']:.6f}s")
                print(f"  Min:   {stats['min']:.6f}s")
                print(f"  Max:   {stats['max']:.6f}s")
    
    def clear(self):
        """Clear all timings."""
        self.timings.clear()
        self.current_timers.clear()

# Global instance
perf_monitor = PerformanceMonitor()

def profile(name: Optional[str] = None):
    """Decorator for profiling functions."""
    def decorator(func):
        profile_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with perf_monitor.timer(profile_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator