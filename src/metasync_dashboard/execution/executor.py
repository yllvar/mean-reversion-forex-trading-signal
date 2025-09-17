"""
Trade execution module for the Mean Reversion strategy.
Currently implements logging only, but can be extended for actual trade execution.
"""
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

from metasync_dashboard.config import LOG_FILE, OUTPUT_DIR

# Set up logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file_path = os.path.join(OUTPUT_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeExecutor:
    """Handles trade execution and logging."""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize the trade executor.
        
        Args:
            paper_trading: If True, only log trades (default: True)
        """
        self.paper_trading = paper_trading
        self.trades_log = os.path.join(OUTPUT_DIR, 'trades.json')
        self.initialize_trades_log()
    
    def initialize_trades_log(self) -> None:
        """Initialize the trades log file if it doesn't exist."""
        if not os.path.exists(self.trades_log):
            with open(self.trades_log, 'w') as f:
                json.dump([], f)
    
    def log_trade(self, symbol: str, signal: str, price: float, 
                 quantity: float = 1.0, reason: str = "") -> None:
        """
        Log a trade to the trade log file.
        
        Args:
            symbol: Trading pair symbol
            signal: Trade signal ('BUY' or 'SELL')
            price: Execution price
            quantity: Trade quantity (default: 1.0)
            reason: Optional reason for the trade (e.g., 'Oversold condition')
        """
        timestamp = datetime.utcnow().isoformat()
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'quantity': quantity,
            'paper_trading': self.paper_trading,
            'reason': reason
        }
        
        # Read existing trades
        try:
            with open(self.trades_log, 'r') as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = []
        
        # Add new trade
        trades.append(trade)
        
        # Write back to file
        with open(self.trades_log, 'w') as f:
            json.dump(trades, f, indent=2)
        
        logger.info(f"Logged {signal} order for {symbol} at {price:.5f}")
    
    def execute_trade(self, symbol: str, signal: str, price: float, 
                     quantity: float = 1.0, reason: str = "") -> None:
        """
        Execute a trade (or log it in paper trading mode).
        
        Args:
            symbol: Trading pair symbol
            signal: Trade signal ('BUY' or 'SELL')
            price: Execution price
            quantity: Trade quantity (default: 1.0)
            reason: Optional reason for the trade
        """
        if signal not in ['BUY', 'SELL']:
            logger.warning(f"Invalid signal: {signal}")
            return
        
        if self.paper_trading:
            logger.info(f"[PAPER] Would {signal} {quantity} {symbol} at {price:.5f}")
            if reason:
                logger.info(f"Reason: {reason}")
        else:
            # TODO: Implement actual trade execution with a broker API
            logger.info(f"[LIVE] Executing {signal} order for {quantity} {symbol} at {price:.5f}")
            if reason:
                logger.info(f"Reason: {reason}")
        
        # Log the trade
        self.log_trade(symbol, signal, price, quantity, reason)

def execute_signals(signals: Dict[str, Tuple[str, float, float]], 
                   paper_trading: bool = True) -> None:
    """
    Execute trades based on generated signals.
    
    Args:
        signals: Dictionary mapping symbols to (signal, price, z_score) tuples
        paper_trading: If True, only log trades (default: True)
    """
    executor = TradeExecutor(paper_trading=paper_trading)
    
    for symbol, (signal, price, z_score) in signals.items():
        if signal == 'HOLD':
            logger.debug(f"No action for {symbol} (z-score: {z_score:.2f})")
            continue
            
        reason = f"Mean reversion signal: z-score = {z_score:.2f}"
        executor.execute_trade(
            symbol=symbol,
            signal=signal,
            price=price,
            quantity=1.0,  # Fixed quantity for now
            reason=reason
        )

if __name__ == "__main__":
    # Example usage
    signals = {
        'EUR/USD': ('BUY', 1.0850, -2.1),
        'GBP/USD': ('SELL', 1.2750, 2.3),
        'USD/JPY': ('HOLD', 109.50, 0.5)
    }
    
    print("=== Paper Trading Mode ===")
    execute_signals(signals, paper_training=True)
    
    print("\n=== Live Trading Mode ===")
    # Uncomment to test live trading (will only log, no actual trades will be placed)
    # execute_signals(signals, paper_trading=False)
