"""
Data fetching and management for market data.
"""
import logging
from typing import Dict, List, Optional, Union
from typing_extensions import Any

import pandas as pd

from .api_client import APIClient
from .cache_manager import CacheManager
from .transformer import DataTransformer
from .. import config

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetches and manages market data with caching support."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize the DataFetcher.
        
        Args:
            api_key: API key for the data provider
            cache_dir: Directory to store cached data
        """
        self.api_key = api_key or config.API_KEY
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.api = APIClient(api_key=self.api_key)
        self.cache = CacheManager(cache_dir=self.cache_dir)
        self.transformer = DataTransformer()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1min",
        exchange: str = "OANDA",
        use_cache: bool = True,
        max_cache_age_hours: int = 24,
        **kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fetch OHLCV data with caching support.
        
        Args:
            symbol: Trading pair symbol (e.g., 'EUR/USD')
            interval: Data interval (1min, 5min, 15min, etc.)
            exchange: Exchange to get data from
            use_cache: Whether to use cached data if available
            max_cache_age_hours: Maximum age of cached data to use (in hours)
            **kwargs: Additional parameters for the API request
            
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize symbol and create cache key
        symbol = self.transformer.normalize_symbol(symbol)
        cache_params = {
            'symbol': symbol,
            'interval': interval,
            'exchange': exchange,
            **kwargs
        }
        
        # Try to get data from cache first
        if use_cache and self.cache.is_cached(cache_params, max_cache_age_hours):
            cached_data = self.cache.get_cached_data(cache_params)
            if cached_data is not None and not cached_data.empty:
                logger.debug(
                    "Using cached data for %s (%s) from %s",
                    symbol, interval, exchange
                )
                return cached_data
        
        # Fetch fresh data from API
        logger.info(
            "Fetching fresh data for %s (%s) from %s",
            symbol, interval, exchange
        )
        try:
            raw_data = self.api.get_ohlcv(
                symbol=symbol,
                interval=interval,
                exchange=exchange,
                **kwargs
            )
            
            # Parse and transform the data
            df = self.transformer.parse_ohlcv(raw_data, symbol, interval)
            
            # Add technical indicators
            df = self.transformer.add_technical_indicators(df)
            
            # Cache the result
            self.cache.cache_data(df, cache_params)
            
            return df
            
        except Exception as e:
            logger.error("Error fetching data for %s: %s", symbol, e)
            # If we have stale cache, return that as fallback
            if use_cache:
                cached_data = self.cache.get_cached_data(cache_params)
                if cached_data is not None and not cached_data.empty:
                    logger.warning(
                        "Using stale cache for %s due to API error",
                        symbol
                    )
                    return cached_data
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1min",
        exchange: str = "OANDA",
        **kwargs: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            interval: Data interval
            exchange: Exchange to get data from
            **kwargs: Additional parameters for the API request
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    exchange=exchange,
                    **kwargs
                )
                results[symbol] = df
            except Exception as e:
                logger.error("Error fetching data for %s: %s", symbol, e)
                results[symbol] = pd.DataFrame()
        return results
    
    def resample_data(
        self,
        df: pd.DataFrame,
        interval: str,
        **kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Resample OHLCV data to a different time frame.
        
        Args:
            df: Input DataFrame with OHLCV data
            interval: Target interval (e.g., '5min', '1h', '1D')
            **kwargs: Additional arguments for resampling
            
        Returns:
            Resampled DataFrame
        """
        return self.transformer.resample_data(df, interval, **kwargs)

# Create a default instance for convenience
data_fetcher = DataFetcher(api_key=config.API_KEY, cache_dir=config.CACHE_DIR)
