"""
Configuration settings for the Mean Reversion Forex Trading System.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

# ========== Directory Configuration ==========
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"  # Directory to store downloaded data
OUTPUT_DIR = BASE_DIR / "output"  # Directory for output files and logs
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ========== API Configuration ==========
API_KEY = "38d9238bd23d4564a2c1f92870b9c9f4"  # Replace with your actual API key
BASE_URL = "https://api.twelvedata.com"
API_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ========== Trading Configuration ==========
# Trading Pairs
SYMBOLS = ["EUR/USD"]
DEFAULT_ACCOUNT_BALANCE = 10_000.0  # USD
RISK_PER_TRADE = 0.01  # 1% of account balance per trade

# ========== Timeframe Configuration ==========
PRIMARY_INTERVAL = "1h"  # For trend analysis
EXECUTION_INTERVAL = "1min"  # For entry/exit timing
HISTORICAL_BACKFILL_DAYS = 30  # Days of historical data to fetch on first run

# Main script configuration (used by main.py)
INTERVAL = EXECUTION_INTERVAL  # Default interval for data fetching
OUTPUT_SIZE = 500  # Default number of data points to fetch

# ========== Rate Limiting ==========
RATE_LIMIT_REQUESTS_PER_MINUTE = 8  # Twelve Data free tier limit
REQUEST_DELAY = 60 / RATE_LIMIT_REQUESTS_PER_MINUTE  # seconds between requests
STAGGER_REQUESTS = True  # Whether to stagger API requests

# ========== Strategy Parameters ==========
# Mean Reversion Parameters
MEAN_WINDOW = 20  # Lookback period for mean calculation (in primary interval)
ZSCORE_ENTRY = 2.0  # Z-score threshold for entry
ZSCORE_EXIT = 0.5  # Z-score threshold for exit

# Risk Management
STOP_LOSS_ATR_MULTIPLIER = 1.5  # ATR multiplier for stop loss
TAKE_PROFIT_ATR_MULTIPLIER = 2.0  # ATR multiplier for take profit
MAX_POSITION_SIZE = 0.1  # Max position size as % of account
MAX_OPEN_TRADES = 3  # Maximum number of open trades at once

# ========== Data Storage ==========
STORAGE_FORMAT = "parquet"  # Options: 'parquet', 'csv', 'feather'
ENABLE_CACHING = True
CACHE_EXPIRY_MINUTES = 5  # Minutes before cached data is considered stale

# ========== Logging Configuration ==========
LOG_FILE = LOG_DIR / "trading_system.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_ROTATION = "1 day"  # Rotate logs daily
LOG_RETENTION = "7 days"  # Keep logs for 7 days

# ========== Execution Mode ==========
TRADING_MODE = "paper"  # Options: 'paper', 'live'
ENABLE_LIVE_TRADING = False  # Must be explicitly enabled for live trading

# ========== Notification Settings ==========
ENABLE_NOTIFICATIONS = True
NOTIFICATION_LEVEL = "trades"  # 'all', 'trades', 'errors'
SLACK_WEBHOOK_URL = None  # Set to your Slack webhook URL for notifications
EMAIL_NOTIFICATIONS = False
EMAIL_RECIPIENTS = []

# ========== Performance Metrics ==========
TRACK_METRICS = True
METRICS_UPDATE_INTERVAL = 300  # seconds (5 minutes)

# ========== Backtesting ==========
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"
BACKTEST_INITIAL_BALANCE = 10_000.0

# ========== Validation ==========
# Validate configuration
assert 0 < RISK_PER_TRADE <= 0.1, "Risk per trade must be between 0 and 0.1 (0-10%)"
assert 0 < MAX_POSITION_SIZE <= 1.0, "Max position size must be between 0 and 1 (0-100%)"
assert TRADING_MODE in ["paper", "live"], "Trading mode must be 'paper' or 'live'"
assert STORAGE_FORMAT in ["parquet", "csv", "feather"], "Invalid storage format"
