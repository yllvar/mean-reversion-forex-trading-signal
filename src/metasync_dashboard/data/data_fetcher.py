"""
Enhanced data fetching module for retrieving OHLCV data from Twelve Data API.

This module provides robust data fetching with rate limiting, caching, and retry logic.
"""
import os
import time
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from metasync_dashboard.config import (
    API_KEY, BASE_URL, PRIMARY_INTERVAL, EXECUTION_INTERVAL, 
    HISTORICAL_BACKFILL_DAYS, DATA_DIR, CACHE_DIR, 
    RATE_LIMIT_REQUESTS_PER_MINUTE, REQUEST_DELAY, STAGGER_REQUESTS,
    API_TIMEOUT, MAX_RETRIES, RETRY_DELAY, STORAGE_FORMAT, ENABLE_CACHING, CACHE_EXPIRY_MINUTES
)
from symbols import get_symbols, normalize_symbol, symbol_to_filename, SymbolFormatError, validate_symbols

# Configure logging
logger = logging.getLogger(__name__)

# Configure requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=RETRY_DELAY,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Rate limiting state
last_request_time = 0
request_counter = 0
rate_limit_reset = 0

def _get_cache_key(symbol: str, interval: str, output_size: int) -> str:
    """
    Generate a cache key for storing/retrieving cached data.
    
    Args:
        symbol: Trading pair symbol (e.g., 'EUR/USD')
        interval: Data interval (e.g., '1min', '1h')
        output_size: Number of data points
        
    Returns:
        str: Cache key string
    """
    try:
        safe_symbol = symbol_to_filename(normalize_symbol(symbol))
        return f"{safe_symbol}_{interval}_{output_size}_{datetime.utcnow().strftime('%Y%m%d')}"
    except SymbolFormatError as e:
        logger.warning(f"Invalid symbol format for cache key: {symbol}, using raw symbol")
        safe_symbol = str(symbol).replace('/', '_')
        return f"{safe_symbol}_{interval}_{output_size}_{datetime.utcnow().strftime('%Y%m%d')}"

def _get_cache_path(symbol: str, interval: str, output_size: int) -> Path:
    """Get the file path for cached data."""
    key = _get_cache_key(symbol, interval, output_size)
    filename = f"{hashlib.md5(key.encode()).hexdigest()}.{STORAGE_FORMAT}"
    return CACHE_DIR / filename

def _load_cached_data(symbol: str, interval: str, output_size: int) -> Optional[pd.DataFrame]:
    """Load cached data if it exists and is not expired."""
    if not ENABLE_CACHING:
        return None
        
    cache_file = _get_cache_path(symbol, interval, output_size)
    
    if not cache_file.exists():
        return None
        
    file_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 60
    
    if file_age > CACHE_EXPIRY_MINUTES:
        logger.debug(f"Cache expired for {symbol} ({file_age:.1f} minutes old)")
        return None
        
    try:
        if STORAGE_FORMAT == 'parquet':
            df = pd.read_parquet(cache_file)
        elif STORAGE_FORMAT == 'csv':
            df = pd.read_csv(cache_file, parse_dates=['datetime'], index_col='datetime')
        elif STORAGE_FORMAT == 'feather':
            df = pd.read_feather(cache_file).set_index('datetime')
        else:
            logger.error(f"Unsupported storage format: {STORAGE_FORMAT}")
            return None
            
        logger.debug(f"Loaded cached data for {symbol} ({len(df)} rows)")
        return df
        
    except Exception as e:
        logger.error(f"Error loading cached data for {symbol}: {str(e)}")
        return None

def _save_to_cache(df: pd.DataFrame, symbol: str, interval: str, output_size: int) -> None:
    """Save data to cache."""
    if not ENABLE_CACHING or df.empty:
        return
        
    try:
        cache_file = _get_cache_path(symbol, interval, output_size)
        
        # Ensure the cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in the specified format
        if STORAGE_FORMAT == 'parquet':
            df.to_parquet(cache_file)
        elif STORAGE_FORMAT == 'csv':
            df.to_csv(cache_file)
        elif STORAGE_FORMAT == 'feather':
            df.reset_index().to_feather(cache_file)
            
        logger.debug(f"Cached data for {symbol} to {cache_file}")
        
    except Exception as e:
        logger.error(f"Error saving cache for {symbol}: {str(e)}")

def _enforce_rate_limit() -> None:
    """Enforce rate limiting with exponential backoff for rate limit errors."""
    global last_request_time, request_counter, rate_limit_reset
    
    current_time = time.time()
    
    # Check if we've hit the rate limit
    if current_time < rate_limit_reset:
        sleep_time = rate_limit_reset - current_time
        logger.warning(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
        time.sleep(sleep_time + 1)  # Add a small buffer
        current_time = time.time()
    
    # Enforce request spacing if enabled
    if STAGGER_REQUESTS and request_counter > 0:
        time_since_last = current_time - last_request_time
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)
    
    last_request_time = current_time
    request_counter += 1

def _make_api_request(url: str, params: dict) -> Optional[dict]:
    """
    Make an API request with rate limiting, retries, and error handling.
    
    Args:
        url: API endpoint URL
        params: Request parameters
        
    Returns:
        JSON response as dict or None if request fails after retries
    """
    global rate_limit_reset, request_counter
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            _enforce_rate_limit()
            
            logger.debug(f"Making request to {url} with params: {params}")
            response = session.get(
                url, 
                params=params, 
                timeout=API_TIMEOUT,
                headers={"User-Agent": "MeanReversionTradingSystem/1.0"}
            )
            
            # Log response status and headers for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Handle rate limiting
            if response.status_code == 429:
                reset_time = int(response.headers.get('x-ratelimit-reset', '60'))
                rate_limit_reset = time.time() + reset_time
                rate_limit = response.headers.get('x-ratelimit-limit', 'Unknown')
                rate_remaining = response.headers.get('x-ratelimit-remaining', 'Unknown')
                logger.warning(
                    f"Rate limited. Status: {response.status_code}, "
                    f"Limit: {rate_limit}, Remaining: {rate_remaining}, "
                    f"Reset in: {reset_time} seconds"
                )
                time.sleep(reset_time + 1)  # Wait for the rate limit to reset
                continue
                
            response.raise_for_status()
            
            # Parse JSON response
            try:
                json_response = response.json()
                logger.debug(f"API response: {json_response}")
                return json_response
            except ValueError as e:
                logger.error(f"Failed to parse JSON response: {e}. Response text: {response.text}")
                raise
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus code: {e.response.status_code}"
                error_msg += f"\nResponse text: {getattr(e.response, 'text', 'No response text')}"
                if e.response.status_code == 429:
                    reset_time = int(e.response.headers.get('x-ratelimit-reset', 60))
                    error_msg += f"\nRate limit reset in: {reset_time} seconds"
            logger.error(error_msg)
            
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            
    logger.error(f"All {MAX_RETRIES} attempts failed for URL: {url}")
    if hasattr(e, 'response') and e.response is not None:
        logger.error(f"Final error response: {e.response.text}")
    return None

def fetch_ohlcv(
    symbol: str, 
    interval: str = EXECUTION_INTERVAL, 
    output_size: int = None,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a given symbol from Twelve Data API with caching and rate limiting.
    
    Args:
        symbol: Trading pair symbol (e.g., 'EUR/USD' or 'EURUSD')
        interval: Data interval (e.g., '1min', '5min', '1h')
        output_size: Number of data points to return (default: None for max)
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with OHLCV data or None if request fails
        
    Raises:
        SymbolFormatError: If the symbol format is invalid
    """
    # Normalize symbol format
    try:
        symbol = normalize_symbol(symbol)
    except Exception as e:
        logger.error(f"Error normalizing symbol {symbol}: {str(e)}")
        return None
        
    # Check cache first
    if use_cache and ENABLE_CACHING:
        try:
            cached_data = _load_cached_data(symbol, interval, output_size or 'max')
            if cached_data is not None:
                logger.debug(f"Using cached data for {symbol} {interval}")
                return cached_data
        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {str(e)}")
    
    # Prepare API request
    url = f"{BASE_URL}/time_series"
    
    # Format symbol for Twelve Data API
    # For forex pairs, we need to use the format like 'EUR/USD' (with forward slash)
    # and also specify the exchange as 'OANDA' for forex data
    formatted_symbol = symbol  # Keep the original format with forward slash
    
    params = {
        "symbol": formatted_symbol,
        "interval": interval,
        "apikey": API_KEY,
        "format": "JSON",
        "exchange": "OANDA"  # Specify OANDA as the exchange for forex data
    }
        
    # Only add output_size if specified
    if output_size:
        params['outputsize'] = min(output_size, 5000)  # API limit
    
    logger.info(f"Fetching {symbol} {interval} data from API...")
    response = _make_api_request(url, params)
    
    if not response:
        logger.error(f"No response received for {symbol}")
        return None
        
    if 'values' not in response:
        logger.error(f"No 'values' in response for {symbol}. Full response: {response}")
        return None
    
    try:
        # Convert to DataFrame and clean up
        df = pd.DataFrame(response['values'])
        
        if df.empty:
            logger.warning(f"Empty data returned for {symbol}")
            return None
        
        # Convert and validate data types
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols + (['volume'] if 'volume' in df.columns else []):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle datetime and set index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # Fill any missing values
        df = df.ffill().dropna()
        
        # Cache the result
        _save_to_cache(df, symbol, interval, output_size or len(df))
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {str(e)}", exc_info=True)
        return None

def fetch_historical_data(
    symbol: str, 
    start_date: str = None, 
    end_date: str = None,
    interval: str = EXECUTION_INTERVAL
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for backtesting or analysis.
    
    Args:
        symbol: Trading pair symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: today)
        interval: Data interval (e.g., '1min', '1h', '1day')
        
    Returns:
        DataFrame with historical OHLCV data or None if request fails
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"
    cache_file = CACHE_DIR / f"{hashlib.md5(cache_key.encode()).hexdigest()}.{STORAGE_FORMAT}"
    
    # Try to load from cache first
    if ENABLE_CACHING and cache_file.exists():
        try:
            if STORAGE_FORMAT == 'parquet':
                return pd.read_parquet(cache_file)
            elif STORAGE_FORMAT == 'csv':
                return pd.read_csv(cache_file, parse_dates=['datetime'], index_col='datetime')
        except Exception as e:
            logger.warning(f"Error loading cached historical data: {str(e)}")
    
    # If not in cache or cache is invalid, fetch from API
    logger.info(f"Fetching historical data for {symbol} ({start_date} to {end_date}, {interval})...")
    
    # Twelve Data API doesn't support direct date range queries in free tier,
    # so we'll fetch the maximum allowed data and filter locally
    df = fetch_ohlcv(symbol, interval=interval, output_size=5000, use_cache=False)
    
    if df is not None:
        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df = df.loc[mask]
        
        # Cache the result
        try:
            if STORAGE_FORMAT == 'parquet':
                df.to_parquet(cache_file)
            elif STORAGE_FORMAT == 'csv':
                df.to_csv(cache_file)
        except Exception as e:
            logger.error(f"Error caching historical data: {str(e)}")
    
    return df

def fetch_all_data(
    symbols: List[str] = None,
    interval: str = EXECUTION_INTERVAL,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols with caching and rate limiting.
    
    Args:
        symbols: List of trading pair symbols (default: from symbols module)
        interval: Data interval (e.g., '1min', '1h')
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary mapping symbols to their respective DataFrames
    """
    if symbols is None:
        symbols = get_symbols()
    
    # Validate all symbols first
    valid_symbols, invalid_symbols = validate_symbols(symbols)
    if invalid_symbols:
        logger.warning(f"Skipping invalid symbols: {', '.join(invalid_symbols)}")
        
    if not valid_symbols:
        logger.error("No valid symbols provided")
        return {}
        
    results = {}
    for symbol in valid_symbols:
        try:
            data = fetch_ohlcv(symbol, interval=interval, use_cache=use_cache)
            if data is not None:
                results[symbol] = data
                
            # Add delay between requests if needed
            if STAGGER_REQUESTS and len(valid_symbols) > 1:
                time.sleep(REQUEST_DELAY)
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
            continue
            
    return results

def save_data(data: Dict[str, pd.DataFrame], interval: str = None) -> None:
    """
    Save data to files in the specified format with error handling.
    
    Args:
        data: Dictionary mapping symbols to DataFrames
        interval: Data interval (e.g., '1min', '1h')
    """
    if not data:
        logger.warning("No data to save")
        return
    
    interval = interval or EXECUTION_INTERVAL
    
    for symbol, df in data.items():
        if df is None or df.empty:
            logger.warning(f"No data to save for {symbol}")
            continue
            
        try:
            # Create a safe filename
            safe_symbol = symbol.replace('/', '_')
            base_filename = f"{safe_symbol}_{interval}"
            
            # Save in the specified format
            if STORAGE_FORMAT == 'parquet':
                filename = DATA_DIR / f"{base_filename}.parquet"
                df.to_parquet(filename)
            elif STORAGE_FORMAT == 'csv':
                filename = DATA_DIR / f"{base_filename}.csv"
                df.to_csv(filename)
            elif STORAGE_FORMAT == 'feather':
                filename = DATA_DIR / f"{base_filename}.feather"
                df.reset_index().to_feather(filename)
            
            logger.info(f"Saved {len(df)} {interval} data points to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}", exc_info=True)
    
    logger.info(f"Successfully saved data for {len(data)} symbols")

if __name__ == "__main__":
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Fetch OHLCV data from Twelve Data API')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='List of symbols to fetch')
    parser.add_argument('--interval', type=str, default=EXECUTION_INTERVAL, 
                        help=f'Data interval (default: {EXECUTION_INTERVAL})')
    parser.add_argument('--output-size', type=int, default=None, 
                        help='Number of data points to fetch (default: max available)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / 'data_fetcher.log')
        ]
    )
    
    logger.info(f"Starting data fetch for {len(args.symbols)} symbols...")
    
    # Fetch and save data
    try:
        data = {}
        for symbol in args.symbols:
            df = fetch_ohlcv(
                symbol=symbol,
                interval=args.interval,
                output_size=args.output_size,
                use_cache=not args.no_cache
            )
            if df is not None:
                data[symbol] = df
        
        if data:
            save_data(data, interval=args.interval)
            logger.info(f"Successfully fetched and saved data for {len(data)} symbols")
        else:
            logger.warning("No data was fetched")
            
    except KeyboardInterrupt:
        logger.info("Data fetch interrupted by user")
    except Exception as e:
        logger.error(f"Error during data fetch: {str(e)}", exc_info=True)
        raise
