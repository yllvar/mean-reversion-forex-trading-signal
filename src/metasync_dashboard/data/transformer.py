"""
Data transformation utilities for processing market data.
"""
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataTransformer:
    """Transforms raw market data into a standardized format."""
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbol format.
        
        Args:
            symbol: Trading pair symbol (e.g., 'EUR/USD' or 'EURUSD')
            
        Returns:
            Normalized symbol with consistent formatting
        """
        # Remove any non-alphanumeric characters and convert to uppercase
        clean = ''.join(c.upper() for c in symbol if c.isalpha())
        
        # Insert '/' in the middle for 6-character pairs (e.g., 'EURUSD' -> 'EUR/USD')
        if len(clean) == 6:
            return f"{clean[:3]}/{clean[3:6]}"
        return symbol.upper()
    
    @staticmethod
    def parse_ohlcv(
        data: Union[Dict, pd.DataFrame], 
        symbol: str,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Parse OHLCV data into a standardized DataFrame.
        
        Args:
            data: Raw OHLCV data from API or cache
            symbol: Trading pair symbol
            interval: Data interval (e.g., '1min', '5min', '1h')
            
        Returns:
            DataFrame with standardized OHLCV columns
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            # Convert API response to DataFrame
            try:
                if 'values' in data:
                    df = pd.DataFrame(data['values'])
                elif 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame(data)
            except Exception as e:
                logger.error("Error parsing OHLCV data: %s", e)
                raise ValueError("Invalid data format") from e
        
        # Standardize column names
        column_map = {
            'datetime': 'timestamp',
            'time': 'timestamp',
            'date': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'value': 'volume',
            'amount': 'volume'
        }
        
        # Rename columns to standard format
        df = df.rename(columns={
            v: k for k, v in column_map.items() 
            if v in df.columns
        })
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = [
            col for col in required_cols 
            if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index and sort
        df = df.set_index('timestamp').sort_index()
        
        # Ensure numeric columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata
        df.attrs['symbol'] = symbol
        df.attrs['interval'] = interval
        
        return df
    
    @staticmethod
    def resample_data(
        df: pd.DataFrame, 
        interval: str,
        ohlc: Optional[List[str]] = None,
        volume: str = 'volume'
    ) -> pd.DataFrame:
        """Resample OHLCV data to a different time frame.
        
        Args:
            df: Input DataFrame with datetime index
            interval: Target interval (e.g., '5min', '1h', '1D')
            ohlc: List of OHLC column names
            volume: Name of the volume column
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
            
        if ohlc is None:
            ohlc = ['open', 'high', 'low', 'close']
            
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Resample OHLC data
        ohlc_dict = {col: 'first' for col in ohlc}
        ohlc_dict['high'] = 'max'
        ohlc_dict['low'] = 'min'
        ohlc_dict['close'] = 'last'
        
        # Resample volume if it exists
        if volume in df.columns:
            ohlc_dict[volume] = 'sum'
        
        # Resample and aggregate
        resampled = df.resample(interval).agg(ohlc_dict)
        
        # Forward fill OHLC values to fill any gaps
        resampled[ohlc] = resampled[ohlc].ffill()
        
        return resampled
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Simple Moving Averages
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume Moving Average
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        return df
