"""
Utility functions for the Mean Reversion Forex Trading System.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from metasync_dashboard.config import OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, create it if it doesn't."""
    os.makedirs(directory, exist_ok=True)

def load_json_file(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The loaded data, or None if the file doesn't exist or is invalid
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load JSON from {filepath}: {e}")
        return None

def save_json_file(data: Any, filepath: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        filepath: Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory(os.path.dirname(filepath) or '.')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except (TypeError, IOError) as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_ohlcv_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from a CSV file.
    
    Args:
        symbol: Trading pair symbol (e.g., 'EUR_USD')
        
    Returns:
        DataFrame with the OHLCV data, or None if the file doesn't exist
    """
    filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}.csv")
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        logger.warning(f"No data file found for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return None

def save_ohlcv_data(df: pd.DataFrame, symbol: str) -> bool:
    """
    Save OHLCV data to a CSV file.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol (e.g., 'EUR/USD')
        
    Returns:
        bool: True if successful, False otherwise
    """
    if df is None or df.empty:
        logger.warning("No data to save")
        return False
        
    filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}.csv")
    try:
        ensure_directory(OUTPUT_DIR)
        df.to_csv(filename)
        logger.info(f"Saved data for {symbol} to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {e}")
        return False

def get_next_run_time(minute: int = 5) -> datetime:
    """
    Calculate the next run time for the scheduler.
    
    Args:
        minute: Minute of the hour to run (default: 5)
        
    Returns:
        datetime: Next run time
    """
    now = datetime.utcnow()
    next_run = now.replace(second=0, microsecond=0) + timedelta(hours=1)
    next_run = next_run.replace(minute=minute)
    
    # If we've already passed the target minute this hour, schedule for next hour
    if next_run < now + timedelta(minutes=1):
        next_run += timedelta(hours=1)
    
    return next_run

def format_timedelta(delta: timedelta) -> str:
    """
    Format a timedelta as a human-readable string.
    
    Args:
        delta: Time delta to format
        
    Returns:
        str: Formatted time delta (e.g., "2h 15m 30s")
    """
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

def validate_config() -> bool:
    """
    Validate the configuration.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    from metasync_dashboard.config import API_KEY, SYMBOLS, INTERVAL, OUTPUT_SIZE
    
    if not API_KEY or API_KEY == "YOUR_TWELVE_DATA_API_KEY":
        logger.error("API key not configured. Please update config.py with your Twelve Data API key.")
        return False
    
    if not SYMBOLS:
        logger.error("No trading symbols configured. Please update config.py with the symbols you want to trade.")
        return False
    
    if not INTERVAL:
        logger.error("No interval configured. Please update config.py with a valid interval.")
        return False
    
    if not OUTPUT_SIZE or OUTPUT_SIZE < 20:  # Minimum lookback for indicators
        logger.error("Output size too small. Please set OUTPUT_SIZE to at least 20 in config.py")
        return False
    
    return True

if __name__ == "__main__":
    # Test the utility functions
    print("Next run time:", get_next_run_time())
    print("Formatted delta:", format_timedelta(timedelta(hours=2, minutes=15, seconds=30)))
    print("Config validation:", validate_config())
