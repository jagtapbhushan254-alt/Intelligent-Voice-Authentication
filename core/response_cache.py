"""
SecureX-Assist - Response Cache
Fast response caching for common queries to improve performance
"""

import logging
import time
from typing import Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for storing command responses to improve performance
    """
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        """
        Initialize response cache
        
        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cached entries (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()  # {query_key: (response, timestamp)}
        self._hits = 0
        self._misses = 0
        logger.info(f"Response cache initialized (max_size={max_size}, ttl={ttl_seconds}s)")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key"""
        return query.lower().strip()
    
    def get(self, query: str) -> Optional[Tuple[bool, str]]:
        """
        Get cached response for a query
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (success, response) if found and not expired, None otherwise
        """
        key = self._normalize_query(query)
        
        if key in self._cache:
            response, timestamp = self._cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Cache HIT for query: '{query}'")
            return response
        
        self._misses += 1
        return None
    
    def set(self, query: str, success: bool, response: str):
        """
        Cache a response
        
        Args:
            query: The user's query
            success: Whether the command was successful
            response: The response text
        """
        key = self._normalize_query(query)
        
        # Add to cache
        self._cache[key] = ((success, response), time.time())
        
        # Move to end
        self._cache.move_to_end(key)
        
        # Remove oldest if exceeded max size
        if len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry: '{oldest_key}'")
        
        logger.debug(f"Cache SET for query: '{query}'")
    
    def clear(self):
        """Clear all cached responses"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Response cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }


# Global cache instance
_response_cache = None


def get_response_cache() -> ResponseCache:
    """Get or create the global response cache instance"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def clear_cache():
    """Clear the global response cache"""
    cache = get_response_cache()
    cache.clear()
