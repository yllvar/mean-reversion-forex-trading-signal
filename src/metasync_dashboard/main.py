"""
Main script for the Mean Reversion Forex Trading System.
Fetches data, generates signals, and executes trades on a schedule.
"""
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import pandas as pd
import schedule

from metasync_dashboard.config import SYMBOLS, INTERVAL, OUTPUT_SIZE
from metasync_dashboard.data.data_fetcher import fetch_all_data, save_data
from metasync_dashboard.strategies.signals import process_signals, format_signal_output
from metasync_dashboard.execution.executor import execute_signals
from metasync_dashboard.utils.utils import get_next_run_time, format_timedelta, validate_config

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set specific log levels for other modules if needed
logging.getLogger('urllib3').setLevel(logging.INFO)  # Reduce noise from urllib3
logging.getLogger('requests').setLevel(logging.INFO)  # Reduce noise from requests

# Enable debug logging for our data fetcher
logging.getLogger('data_fetcher').setLevel(logging.DEBUG)

class TradingBot:
    """Main trading bot class that orchestrates the trading system."""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize the trading bot.
        
        Args:
            paper_trading: If True, only log trades (default: True)
        """
        self.paper_trading = paper_trading
        self.last_run = None
        self.run_count = 0
    
    def run_strategy(self) -> None:
        """Run the mean reversion strategy on all configured symbols."""
        self.run_count += 1
        start_time = time.time()
        current_time = datetime.now(timezone.utc)
        logger.info(f"\n{'='*50}")
        logger.info(f"Running strategy (Run #{self.run_count}) - {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        try:
            # Print cycle start information
            print(f"\n=== Cycle Started at {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC ===")
            print(f"Processing {len(SYMBOLS)} symbol(s): {', '.join(SYMBOLS)}")
            
            # Step 1: Fetch data for all symbols
            logger.info(f"Fetching {INTERVAL} data for {len(SYMBOLS)} symbols...")
            data = fetch_all_data(SYMBOLS)
            
            if not data:
                logger.warning("No data was fetched. Check your API key and internet connection.")
                return
                
            # Print price information for each symbol
            for symbol, df in data.items():
                if not df.empty:
                    last_row = df.iloc[-1]
                    print(f"{symbol}: Price={last_row['close']:.5f} at {last_row.name.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
            # Step 2: Save the data for later analysis
            save_data(data)
            
            # Step 3: Generate trading signals
            logger.info("Generating trading signals...")
            signals = process_signals(data)
            
            # Step 4: Print signals
            print("\n=== Trading Signals ===")
            print(format_signal_output(signals))
            
            # Step 5: Execute trades based on signals
            if signals:
                execute_signals(signals, paper_trading=self.paper_trading)
            
            self.last_run = current_time
            logger.info(f"Strategy completed in {time.time() - start_time:.2f} seconds")
            print(f"\n=== Cycle Completed in {time.time() - start_time:.2f} seconds ===\n")
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}", exc_info=True)
            print(f"\n=== Error in cycle: {str(e)} ===\n")
            raise
    
    def schedule_next_run(self, minute: int = 5) -> None:
        """
        Schedule the next run of the strategy.
        
        Args:
            minute: Minute of the hour to run (default: 5 minutes past the hour)
        """
        # Clear any existing schedules
        schedule.clear()
        
        # Schedule the job to run every minute
        schedule.every().minute.at(":00").do(self.run_strategy)
        
        # Calculate next run time (next minute)
        now = datetime.now(timezone.utc)
        next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        wait_seconds = (next_run - now).total_seconds()
        
        logger.info("Scheduled to run every minute")
        logger.info(f"Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {format_timedelta(timedelta(seconds=wait_seconds))})")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mean Reversion Forex Trading Bot')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper trading)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--minute', type=int, default=5, help='Minute of the hour to run (default: 5)')
    return parser.parse_args()

def main() -> None:
    """Main entry point for the trading bot."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate configuration
    if not validate_config():
        return
    
    # Initialize the trading bot
    bot = TradingBot(paper_trading=not args.live)
    
    # Run once immediately if requested
    if args.once:
        bot.run_strategy()
        return
    
    # Otherwise, schedule the bot to run periodically
    bot.schedule_next_run(minute=args.minute)
    
    try:
        # Run the scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Trading bot stopped")

if __name__ == "__main__":
    main()
