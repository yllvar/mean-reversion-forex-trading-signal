"""
Cache manager for storing and retrieving market data.
"""
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from pyarrow import parquet as pq
from pyarrow import Table

from .. import config

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of market data to improve performance."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cached files. Defaults to settings.CACHE_DIR.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a unique cache key from request parameters.
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            MD5 hash of the parameters as a string
        """
        # Sort parameters to ensure consistent key generation
        sorted_params = json.dumps(params, sort_keys=True).encode('utf-8')
        return hashlib.md5(sorted_params).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the full path to a cache file.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}.parquet"
    
    def is_cached(self, params: Dict[str, Any], max_age_hours: int = 24) -> bool:
        """Check if data is in the cache and not expired.
        
        Args:
            params: Request parameters used to generate cache key
            max_age_hours: Maximum age of cache in hours before considering it stale
            
        Returns:
            True if valid cached data exists, False otherwise
        """
        key = self._get_cache_key(params)
        cache_file = self._get_cache_path(key)
        
        if not cache_file.exists():
            return False
            
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return cache_age < timedelta(hours=max_age_hours)
    
    def get_cached_data(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get data from cache.
        
        Args:
            params: Request parameters used to generate cache key
            
        Returns:
            Cached DataFrame if found and valid, None otherwise
        """
        key = self._get_cache_key(params)
        cache_file = self._get_cache_path(key)
        
        if not cache_file.exists():
            return None
            
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning("Error reading cache file %s: %s", cache_file, e)
            return None
    
    def cache_data(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]], 
        params: Dict[str, Any]
    ) -> Path:
        """Cache data to disk.
        
        Args:
            data: Data to cache (DataFrame or dict)
            params: Request parameters used to generate cache key
            
        Returns:
            Path to the cached file
        """
        if isinstance(data, dict):
            # Convert dict to DataFrame if needed
            df = pd.DataFrame(data)
        else:
            df = data
            
        key = self._get_cache_key(params)
        cache_file = self._get_cache_path(key)
        
        try:
            # Convert DataFrame to PyArrow Table and write to Parquet
            table = Table.from_pandas(df)
            pq.write_table(table, cache_file)
            logger.debug("Cached data to %s", cache_file)
            return cache_file
        except Exception as e:
            logger.error("Error caching data to %s: %s", cache_file, e)
            raise
    
    def clear_old_cache(self, max_age_days: int = 7) -> int:
        """Remove cache files older than the specified number of days.
        
        Args:
            max_age_days: Maximum age of cache files in days
            
        Returns:
            Number of cache files removed
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    cache_file.unlink()
                    removed += 1
                    logger.debug("Removed old cache file: %s", cache_file)
                except Exception as e:
                    logger.warning("Error removing cache file %s: %s", cache_file, e)
        
        logger.info("Cleaned up %d old cache files", removed)
        return removed
