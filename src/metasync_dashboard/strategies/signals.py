"""
Enhanced Signal Generation Module for Multi-Timeframe Mean Reversion Strategy.

This module implements a sophisticated mean reversion strategy that incorporates:
- Multiple timeframe analysis (1m for entries, 1h for trend)
- Volume profile analysis
- Volatility-adjusted position sizing
- Risk management with ATR-based stops
- Confirmation indicators (RSI, MACD)
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from metasync_dashboard.config import (
    MEAN_WINDOW, ZSCORE_ENTRY, ZSCORE_EXIT, 
    STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_ATR_MULTIPLIER,
    PRIMARY_INTERVAL, EXECUTION_INTERVAL, MAX_POSITION_SIZE,
    RISK_PER_TRADE, LOG_LEVEL, TRADING_MODE
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

@dataclass
class TradeSignal:
    """Container for trade signal details."""
    symbol: str
    signal: str  # 'BUY', 'SELL', or 'HOLD'
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size: Optional[float] = None
    confidence: float = 0.0
    indicators: Optional[dict] = None
    timestamp: datetime = None
    
    def to_dict(self) -> dict:
        """Convert signal to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'size': self.size,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'indicators': self.indicators or {}
        }

class MeanReversionStrategy:
    """
    Enhanced Mean Reversion Strategy with multiple timeframes and risk management.
    
    This strategy combines:
    - Primary timeframe (1h) for trend direction
    - Execution timeframe (1m) for precise entries
    - Volume profile confirmation
    - Volatility-based position sizing
    - ATR-based stop loss and take profit
    """
    
    def __init__(self, 
                 primary_interval: str = PRIMARY_INTERVAL,
                 execution_interval: str = EXECUTION_INTERVAL,
                 lookback_periods: int = MEAN_WINDOW,
                 zscore_entry: float = ZSCORE_ENTRY,
                 zscore_exit: float = ZSCORE_EXIT,
                 stop_loss_atr: float = STOP_LOSS_ATR_MULTIPLIER,
                 take_profit_atr: float = TAKE_PROFIT_ATR_MULTIPLIER):
        """
        Initialize the enhanced mean reversion strategy.
        
        Args:
            primary_interval: Primary timeframe for trend analysis (e.g., '1h')
            execution_interval: Timeframe for execution (e.g., '1m')
            lookback_periods: Number of periods for mean calculation
            zscore_entry: Z-score threshold for entry signals
            zscore_exit: Z-score threshold for exit signals
            stop_loss_atr: ATR multiplier for stop loss
            take_profit_atr: ATR multiplier for take profit
        """
        self.primary_interval = primary_interval
        self.execution_interval = execution_interval
        self.lookback_periods = lookback_periods
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.min_volume_periods = 20  # Minimum periods for volume analysis
        
        # State for tracking
        self.last_signal = None
        self.position_size = 0.0
        self.current_risk = 0.0
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the enhanced mean reversion strategy.
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            
        Returns:
            pd.DataFrame: DataFrame with added indicator columns
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
            
        if len(df) < self.lookback_periods:
            logger.warning(f"Insufficient data points. Have {len(df)}, need at least {self.lookback_periods}")
            return df
            
        try:
            # Make a working copy
            df = df.copy()
            
            # Ensure numeric data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            if len(df) < self.lookback_periods:
                logger.warning(f"Not enough valid data points after cleaning. Have {len(df)}, need {self.lookback_periods}")
                return df
            
            # 1. Calculate basic statistics
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 2. Mean Reversion Indicators
            df['sma'] = df['close'].rolling(window=self.lookback_periods).mean()
            df['std'] = df['close'].rolling(window=self.lookback_periods).std()
            df['z_score'] = (df['close'] - df['sma']) / df['std'].replace(0, 1e-9)
            
            # Handle volume if available, otherwise use a default value
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=self.min_volume_periods).mean()
            else:
                # If volume data is not available, create a dummy column with high enough values
                df['volume_sma'] = float('inf')  # This will always pass the volume check
            
            # 3. Volatility (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 4. Volume Analysis
            if 'volume' in df.columns:
                df['volume_zscore'] = (df['volume'] - df['volume_sma']) / \
                                    df['volume'].rolling(window=self.min_volume_periods).std().replace(0, 1e-9)
            else:
                df['volume_zscore'] = 0.0  # Default to neutral if no volume data
            
            # 5. Additional Confirmation Indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-9)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # 6. Calculate potential stop loss and take profit levels
            df['stop_loss_long'] = df['low'].rolling(window=5).min() - (df['atr'] * self.stop_loss_atr)
            df['take_profit_long'] = df['close'] + (df['atr'] * self.take_profit_atr)
            df['stop_loss_short'] = df['high'].rolling(window=5).max() + (df['atr'] * self.stop_loss_atr)
            df['take_profit_short'] = df['close'] - (df['atr'] * self.take_profit_atr)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return df
    
    def generate_signal(self, df: pd.DataFrame, account_balance: float = None) -> TradeSignal:
        """
        Generate trading signals with enhanced logic.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            account_balance: Current account balance for position sizing (optional)
            
        Returns:
            TradeSignal: Object containing signal details
        """
        # Initialize default signal
        symbol = df.attrs.get('symbol', 'UNKNOWN')
        signal = TradeSignal(
            symbol=symbol,
            signal='HOLD',
            price=df['close'].iloc[-1],
            timestamp=df.index[-1]
        )
        
        # Check if we have enough data
        required_columns = ['close', 'z_score', 'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal']
        if df.empty or not all(col in df.columns for col in required_columns):
            logger.warning("Missing required columns for signal generation")
            return signal
            
        # Get the most recent data
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        # Update signal price
        signal.price = last['close']
        
        # 1. Check primary trend (using higher timeframe data if available)
        primary_trend = self._get_primary_trend(df)
        
        # 2. Check volume confirmation
        volume_confirmed = self._check_volume_confirmation(df)
        
        # 3. Check RSI for overbought/oversold
        rsi_signal = self._get_rsi_signal(last, prev)
        
        # 4. Check MACD for momentum
        macd_signal = self._get_macd_signal(last, prev)
        
        # 5. Check z-score for mean reversion
        zscore_signal = self._get_zscore_signal(last, prev)
        
        # 6. Calculate position size based on risk
        position_size = self._calculate_position_size(df, account_balance) if account_balance else None
        
        # 7. Determine final signal with confidence score
        signal.indicators = {
            'primary_trend': primary_trend,
            'volume_confirmed': volume_confirmed,
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal'],
            'z_score': last['z_score'],
            'atr': last['atr']
        }
        
        # Generate signals based on conditions
        if (zscore_signal == 'BUY' and 
            primary_trend != 'downtrend' and 
            volume_confirmed and 
            rsi_signal in ['oversold', 'neutral'] and
            macd_signal in ['bullish', 'neutral']):
            
            signal.signal = 'BUY'
            signal.stop_loss = last['stop_loss_long']
            signal.take_profit = last['take_profit_long']
            signal.size = position_size
            signal.confidence = self._calculate_confidence(
                z_score=last['z_score'],
                rsi=last['rsi'],
                volume_zscore=last.get('volume_zscore', 0),
                trend_strength=1.0 if primary_trend == 'uptrend' else 0.5
            )
            
        elif (zscore_signal == 'SELL' and 
              primary_trend != 'uptrend' and 
              volume_confirmed and 
              rsi_signal in ['overbought', 'neutral'] and
              macd_signal in ['bearish', 'neutral']):
            
            signal.signal = 'SELL'
            signal.stop_loss = last['stop_loss_short']
            signal.take_profit = last['take_profit_short']
            signal.size = position_size
            signal.confidence = self._calculate_confidence(
                z_score=last['z_score'],
                rsi=last['rsi'],
                volume_zscore=last.get('volume_zscore', 0),
                trend_strength=1.0 if primary_trend == 'downtrend' else 0.5
            )
        
        return signal
    
    def _get_primary_trend(self, df: pd.DataFrame) -> str:
        """Determine the primary trend direction."""
        if len(df) < 50:  # Minimum data points for trend
            return 'neutral'
            
        # Use SMA crossover for trend
        sma_fast = df['close'].rolling(window=20).mean()
        sma_slow = df['close'].rolling(window=50).mean()
        
        if sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-5] <= sma_slow.iloc[-5]:
            return 'uptrend'
        elif sma_fast.iloc[-1] < sma_slow.iloc[-1] and sma_fast.iloc[-5] >= sma_slow.iloc[-5]:
            return 'downtrend'
        elif sma_fast.iloc[-1] > sma_slow.iloc[-1]:
            return 'weak_uptrend'
        elif sma_fast.iloc[-1] < sma_slow.iloc[-1]:
            return 'weak_downtrend'
        return 'neutral'
    
    def _check_volume_confirmation(self, df: pd.DataFrame, window: int = 20) -> bool:
        """Check if volume confirms the price movement."""
        if 'volume' not in df.columns or len(df) < window:
            return True  # Default to True if we can't determine
            
        # Check if current volume is above average
        avg_volume = df['volume'].rolling(window=window).mean().iloc[-1]
        return df['volume'].iloc[-1] > avg_volume * 0.8  # 80% of average is acceptable
    
    def _get_rsi_signal(self, last: pd.Series, prev: pd.Series) -> str:
        """Get RSI-based signal."""
        if 'rsi' not in last or pd.isna(last['rsi']):
            return 'neutral'
            
        if last['rsi'] < 30:
            return 'oversold'
        elif last['rsi'] > 70:
            return 'overbought'
        elif 30 <= last['rsi'] <= 70:
            return 'neutral'
        else:
            return 'extreme'
    
    def _get_macd_signal(self, last: pd.Series, prev: pd.Series) -> str:
        """Get MACD-based signal."""
        if 'macd' not in last or 'macd_signal' not in last:
            return 'neutral'
            
        # MACD line crosses above signal line
        if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            return 'bullish'
        # MACD line crosses below signal line
        elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            return 'bearish'
        # MACD above signal line
        elif last['macd'] > last['macd_signal']:
            return 'weak_bullish'
        # MACD below signal line
        elif last['macd'] < last['macd_signal']:
            return 'weak_bearish'
        return 'neutral'
    
    def _get_zscore_signal(self, last: pd.Series, prev: pd.Series) -> str:
        """Get z-score based signal."""
        if 'z_score' not in last or pd.isna(last['z_score']):
            return 'hold'
            
        # Entry signals
        if last['z_score'] <= -self.zscore_entry and prev['z_score'] > -self.zscore_entry:
            return 'BUY'
        elif last['z_score'] >= self.zscore_entry and prev['z_score'] < self.zscore_entry:
            return 'SELL'
        # Exit signals
        elif (last['z_score'] >= -self.zscore_exit and 
              last['z_score'] <= self.zscore_exit and 
              (prev['z_score'] < -self.zscore_exit or prev['z_score'] > self.zscore_exit)):
            return 'EXIT'
        return 'HOLD'
    
    def _calculate_position_size(self, df: pd.DataFrame, account_balance: float) -> float:
        """Calculate position size based on risk parameters."""
        if account_balance is None or 'atr' not in df.columns or df['atr'].isna().all():
            return 0.0
            
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr']
        
        if price <= 0 or atr <= 0:
            return 0.0
            
        # Risk per trade as a dollar amount
        risk_amount = account_balance * RISK_PER_TRADE
        
        # Position size based on ATR-based stop loss
        stop_distance = atr * self.stop_loss_atr
        if stop_distance > 0:
            position_size = (risk_amount / stop_distance) / price
            # Apply maximum position size
            max_size = (account_balance * MAX_POSITION_SIZE) / price
            return min(position_size, max_size)
        return 0.0
    
    def _calculate_confidence(self, z_score: float, rsi: float, 
                            volume_zscore: float, trend_strength: float) -> float:
        """Calculate confidence score for the signal (0-1)."""
        # Base confidence from z-score (further from mean = higher confidence)
        z_confidence = min(abs(z_score) / self.zscore_entry, 1.0)
        
        # RSI confidence (more extreme RSI = higher confidence in mean reversion)
        rsi_confidence = 0.0
        if rsi < 30 or rsi > 70:  # In overbought/oversold territory
            rsi_confidence = min((abs(rsi - 50) / 20), 1.0)  # 0-1 based on distance from 50
        
        # Volume confidence (higher volume = higher confidence)
        vol_confidence = min(max(volume_zscore + 1, 0), 2) / 2  # Normalize to 0-1
        
        # Combine confidences with weights
        confidence = (
            0.4 * z_confidence + 
            0.3 * rsi_confidence + 
            0.2 * vol_confidence + 
            0.1 * trend_strength
        )
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _check_price_trend(self, df: pd.DataFrame) -> str:
        """
        Check the current price trend using multiple timeframes and indicators.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            str: 'uptrend', 'downtrend', or 'neutral'
        """
        if len(df) < 50:  # Minimum data points for reliable trend
            return 'neutral'
            
        # 1. Check multiple moving averages
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        # 2. Check MACD histogram
        macd_hist = df.get('macd_hist', df['macd'] - df['macd_signal'] if 'macd' in df.columns else None)
        
        # 3. Check ADX for trend strength
        adx = self._calculate_adx(df) if len(df) > 14 else 0
        
        # Determine trend based on multiple factors
        ma_bullish = sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]
        ma_bearish = sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]
        
        macd_bullish = macd_hist is not None and macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]
        macd_bearish = macd_hist is not None and macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]
        
        # Strong trend requires multiple confirmations
        if ma_bullish and (macd_bullish or adx > 25):
            return 'uptrend'
        elif ma_bearish and (macd_bearish or adx > 25):
            return 'downtrend'
        elif (sma_20.iloc[-1] > sma_50.iloc[-1]) and macd_bullish:
            return 'weak_uptrend'
        elif (sma_20.iloc[-1] < sma_50.iloc[-1]) and macd_bearish:
            return 'weak_downtrend'
        return 'neutral'
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate Average Directional Index (ADX) for trend strength."""
        if len(df) < window * 2:  # Need enough data for reliable ADX
            return 0.0
            
        try:
            high, low, close = df['high'], df['low'], df['close']
            
            # Calculate +DM and -DM
            up = high.diff()
            down = low.diff() * -1
            
            plus_dm = up.where((up > down) & (up > 0), 0)
            minus_dm = down.where((down > up) & (down > 0), 0)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the values
            alpha = 1.0 / window
            tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate directional indicators
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
            adx = dx.rolling(window=window).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating ADX: {str(e)}")
            return 0.0

def process_signals(
    data: Dict[str, pd.DataFrame], 
    strategy: Optional[MeanReversionStrategy] = None,
    account_balance: Optional[float] = None
) -> Dict[str, tuple]:
    """
    Process signals for multiple symbols with enhanced features.
    
    Args:
        data: Dictionary mapping symbols to their DataFrames
        strategy: Optional strategy instance (creates one if None)
        account_balance: Current account balance for position sizing
        
    Returns:
        Dict[str, tuple]: Dictionary mapping symbols to (signal, price, z_score) tuples
    """
    if not data:
        logger.warning("No data provided for signal processing")
        return {}
    
    # Initialize strategy if not provided
    if strategy is None:
        strategy = MeanReversionStrategy()
    
    signals = {}
    
    for symbol, df in data.items():
        try:
            # Add symbol to DataFrame attributes for reference
            df.attrs['symbol'] = symbol
            
            # Calculate indicators if not already present
            required_indicators = ['z_score', 'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal']
            if not all(indicator in df.columns for indicator in required_indicators):
                df = strategy.calculate_indicators(df)
            
            # Generate signal
            signal_obj = strategy.generate_signal(df, account_balance)
            
            # Convert TradeSignal to tuple (signal, price, z_score)
            z_score = df['z_score'].iloc[-1] if not df.empty else 0
            signals[symbol] = (signal_obj.signal, signal_obj.price, z_score)
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {str(e)}", exc_info=True)
            price = df['close'].iloc[-1] if not df.empty and 'close' in df.columns else 0
            signals[symbol] = ('HOLD', price, 0)
    
    return signals

def format_signal_output(signals: Dict[str, tuple], detailed: bool = False) -> str:
    """
    Format signals into a human-readable string with optional details.
    
    Args:
        signals: Dictionary of (signal, price, z_score) tuples
        detailed: Whether to include detailed indicator values (not fully implemented for tuple format)
        
    Returns:
        str: Formatted string with signals
    """
    if not signals:
        return "No signals generated."
    
    output = ["\n=== Trading Signals ===\n"]
    
    for symbol, (signal_type, price, z_score) in signals.items():
        signal_str = [
            f"{symbol}: {signal_type} at {price:.5f}",
            f"  Z-Score: {z_score:.2f}"
        ]
        
        if detailed:
            signal_str.append("  [Detailed indicators not available in tuple format]")
        
        output.append("\n".join(signal_str))
    
    return "\n".join(output)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate trading signals using Mean Reversion Strategy')
    parser.add_argument('--symbols', nargs='+', default=['EUR/USD', 'GBP/USD', 'USD/JPY'],
                       help='List of symbols to process')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing price data files')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Account balance for position sizing')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed indicator values')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path('logs') / 'signals.log')
        ]
    )
    
    # Example usage with sample data
    try:
        # In a real implementation, you would load actual market data here
        # For demonstration, we'll generate some sample data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        base_prices = 100 + np.cumsum(np.random.randn(500) * 0.1)
        
        # Create sample data for each symbol
        data = {}
        for i, symbol in enumerate(args.symbols):
            # Generate slightly different price series for each symbol
            prices = base_prices * (1 + (i * 0.05) + (np.random.randn(500) * 0.01))
            
            df = pd.DataFrame({
                'open': prices - np.abs(np.random.randn(500) * 0.1),
                'high': prices + np.abs(np.random.randn(500) * 0.1),
                'low': prices - np.abs(np.random.randn(500) * 0.15),
                'close': prices,
                'volume': np.random.randint(100, 10000, 500)
            }, index=dates)
            
            # Store with symbol in attributes
            df.attrs['symbol'] = symbol
            data[symbol] = df
        
        # Initialize strategy
        strategy = MeanReversionStrategy()
        
        # Process signals
        signals = process_signals(data, strategy, args.balance)
        
        # Display results
        print(format_signal_output(signals, detailed=args.detailed))
        
    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}", exc_info=True)
        raise
