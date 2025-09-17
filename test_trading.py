"""
Test script for the 1-minute trading system.
"""
import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_SYMBOLS = ['EUR/USD', 'USD/JPY']  # Limit symbols for testing
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def test_data_fetching() -> bool:
    """Test data fetching functionality with rate limiting and caching."""
    from data_fetcher import fetch_ohlcv, fetch_all_data, save_data, load_data
    
    logger.info("Testing data fetching...")
    
    # Test 1: Single symbol fetch
    logger.info("\n--- Testing single symbol fetch ---")
    symbol = TEST_SYMBOLS[0]
    df = fetch_ohlcv(symbol, output_size=5)  # Only fetch 5 data points for testing
    
    if df is None or df.empty:
        logger.error("Failed to fetch single symbol data")
        return False
    
    logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
    logger.info(f"Data sample:\n{df.head()}")
    
    # Test 2: Fetch all symbols with rate limiting
    logger.info("\n--- Testing fetch_all_data with rate limiting ---")
    data = fetch_all_data(use_cached=False)
    
    if not data:
        logger.error("Failed to fetch data for any symbols")
        return False
    
    logger.info(f"Successfully fetched data for {len(data)} symbols")
    for symbol, df in data.items():
        logger.info(f"{symbol}: {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    # Test 3: Test caching
    logger.info("\n--- Testing data caching ---")
    cache_file = os.path.join(TEST_DATA_DIR, f"{symbol.replace('/', '_')}_1min.csv")
    if os.path.exists(cache_file):
        os.remove(cache_file)  # Clear any existing cache
    
    # First fetch (should hit API)
    start_time = time.time()
    df1 = fetch_ohlcv(symbol, output_size=5)
    api_fetch_time = time.time() - start_time
    
    # Second fetch (should use cache)
    start_time = time.time()
    df2 = fetch_ohlcv(symbol, output_size=5)
    cache_fetch_time = time.time() - start_time
    
    logger.info(f"API fetch time: {api_fetch_time:.3f}s, Cache fetch time: {cache_fetch_time:.3f}s")
    
    if not os.path.exists(cache_file):
        logger.error("Cache file was not created")
        return False
    
    # Test 4: Load from cache
    logger.info("\n--- Testing data loading from cache ---")
    cached_data = load_data()
    if not cached_data:
        logger.warning("No data loaded from cache")
    else:
        logger.info(f"Loaded {len(cached_data)} symbols from cache")
    
    return True

def test_signal_generation() -> bool:
    """Test signal generation with various market conditions."""
    from signals import MeanReversionStrategy, process_signals, format_signal_output
    import numpy as np
    
    logger.info("\nTesting signal generation...")
    
    # Test 1: Test with sample data
    logger.info("\n--- Testing with sample data ---")
    
    # Create test data with a clear mean reversion pattern
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
    base_price = 1.1000
    
    # Create a mean-reverting series
    returns = np.random.normal(0, 0.001, 99)
    prices = [base_price]
    for r in returns:
        # Add some mean reversion
        deviation = (prices[-1] - base_price) / base_price
        mean_reversion = -0.5 * deviation  # Mean reversion factor
        prices.append(prices[-1] * (1 + r + mean_reversion))
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.0002 for p in prices],
        'low': [p * 0.9998 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    }, index=dates)
    
    # Initialize strategy
    strategy = MeanReversionStrategy()
    
    # Test indicator calculation
    df_with_indicators = strategy.calculate_indicators(test_data)
    if df_with_indicators is None or df_with_indicators.empty:
        logger.error("Failed to calculate indicators")
        return False
    
    logger.info("Successfully calculated indicators")
    logger.info(f"Indicators sample:\n{df_with_indicators[['close', 'sma', 'upper_band', 'lower_band']].tail()}")
    
    # Test signal generation
    signal, price = strategy.generate_signal(df_with_indicators)
    logger.info(f"Generated signal: {signal} at price {price:.5f}")
    
    # Test 2: Test with real data (if available)
    try:
        from data_fetcher import load_data
        logger.info("\n--- Testing with real data ---")
        
        # Try to load cached data first
        data = load_data()
        if not data:
            logger.warning("No cached data available for signal testing")
            return True  # Not a failure, just skip this part
        
        # Process signals for all available data
        signals = process_signals(data)
        
        if not signals:
            logger.warning("No signals generated from real data")
            return True  # Not necessarily a failure
        
        # Print formatted signals
        print("\n" + "="*50)
        print("Trading Signals")
        print("="*50)
        print(format_signal_output(signals))
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error testing with real data: {str(e)}")
        return False
    
    return True

def test_trade_execution() -> bool:
    """Test trade execution in paper trading mode with various scenarios."""
    from executor import execute_signals, PaperTradingAccount, save_trades_to_json
    from signals import generate_signals
    import json
    
    logger.info("\nTesting trade execution...")
    
    # Test 1: Initialize paper trading account
    logger.info("\n--- Testing paper trading account ---")
    account = PaperTradingAccount(initial_balance=10000.0)
    
    # Test account operations
    logger.info(f"Initial balance: ${account.get_balance():.2f}")
    
    # Test 2: Execute sample trades
    logger.info("\n--- Testing trade execution ---")
    test_signals = [
        {'symbol': 'EUR/USD', 'action': 'BUY', 'price': 1.1000, 'timestamp': '2023-01-01 12:00:00'},
        {'symbol': 'USD/JPY', 'action': 'SELL', 'price': 150.50, 'timestamp': '2023-01-01 12:01:00'},
        {'symbol': 'EUR/USD', 'action': 'SELL', 'price': 1.1050, 'timestamp': '2023-01-01 12:30:00'},
        {'symbol': 'USD/JPY', 'action': 'BUY', 'price': 150.00, 'timestamp': '2023-01-01 13:00:00'},
    ]
    
    for signal in test_signals:
        logger.info(f"Executing {signal['action']} {signal['symbol']} at {signal['price']}")
        account.execute_trade(
            symbol=signal['symbol'],
            action=signal['action'],
            price=signal['price'],
            timestamp=pd.Timestamp(signal['timestamp'])
        )
        logger.info(f"Current balance: ${account.get_balance():.2f}, Open positions: {len(account.positions)}")
    
    # Test 3: Save trades to JSON
    logger.info("\n--- Testing trade logging ---")
    test_log_file = os.path.join(TEST_DATA_DIR, 'test_trades.json')
    if os.path.exists(test_log_file):
        os.remove(test_log_file)
    
    # Save test trades
    save_trades_to_json(account.trade_history, test_log_file)
    
    if not os.path.exists(test_log_file):
        logger.error("Failed to save trade log")
        return False
    
    # Verify trade log
    with open(test_log_file, 'r') as f:
        trades = json.load(f)
        logger.info(f"Saved {len(trades)} trades to {test_log_file}")
    
    # Test 4: Test with real signals (if available)
    try:
        from data_fetcher import load_data
        from signals import process_signals
        
        logger.info("\n--- Testing with real signals ---")
        data = load_data()
        if not data:
            logger.warning("No data available for trade execution testing")
            return True  # Not a failure, just skip this part
        
        signals = process_signals(data)
        if not signals:
            logger.warning("No signals generated for trade execution testing")
            return True  # Not necessarily a failure
        
        # Create a new account for this test
        test_account = PaperTradingAccount(initial_balance=10000.0)
        execute_signals(signals, account=test_account, paper_trading=True)
        
        logger.info(f"Final balance after executing {len(signals)} signals: ${test_account.get_balance():.2f}")
        
    except Exception as e:
        logger.error(f"Error testing with real signals: {str(e)}")
        return False
    
    return True

def run_all_tests() -> int:
    """Run all test cases with detailed reporting."""
    import time
    from datetime import datetime
    
    # Set up test results
    test_results = {
        'start_time': datetime.now().isoformat(),
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_dir': os.getcwd()
        },
        'tests': []
    }
    
    # Define test cases
    test_cases = [
        ("Data Fetching", test_data_fetching, "Verifies data fetching, caching, and rate limiting"),
        ("Signal Generation", test_signal_generation, "Tests signal generation with both synthetic and real data"),
        ("Trade Execution", test_trade_execution, "Validates trade execution and paper trading functionality")
    ]
    
    logger.info("\n" + "="*50)
    logger.info("Starting 1-minute Trading System Tests")
    logger.info("="*50)
    logger.info(f"Start Time: {test_results['start_time']}")
    logger.info(f"Python: {test_results['environment']['python_version']}")
    logger.info(f"Platform: {test_results['environment']['platform']}")
    logger.info(f"Working Directory: {test_results['environment']['working_dir']}")
    
    # Run tests
    for name, test_func, description in test_cases:
        logger.info("\n" + "="*50)
        logger.info(f"Running test: {name}")
        logger.info("-" * 50)
        logger.info(f"Description: {description}")
        
        test_result = {
            'name': name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'status': 'RUNNING',
            'duration': None,
            'error': None
        }
        
        start_time = time.time()
        try:
            success = test_func()
            test_result['status'] = 'PASSED' if success else 'FAILED'
        except Exception as e:
            logger.error(f"Test error: {str(e)}", exc_info=True)
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
        
        # Calculate duration
        duration = time.time() - start_time
        test_result['duration'] = f"{duration:.2f}s"
        test_result['end_time'] = datetime.now().isoformat()
        
        # Log test result
        status_emoji = '✅' if test_result['status'] == 'PASSED' else '❌'
        logger.info(f"\n{status_emoji} {name}: {test_result['status']} in {test_result['duration']}")
        
        test_results['tests'].append(test_result)
    
    # Calculate summary
    test_results['end_time'] = datetime.now().isoformat()
    total_duration = (datetime.fromisoformat(test_results['end_time']) - 
                     datetime.fromisoformat(test_results['start_time'])).total_seconds()
    
    passed = sum(1 for t in test_results['tests'] if t['status'] == 'PASSED')
    failed = sum(1 for t in test_results['tests'] if t['status'] == 'FAILED')
    errors = sum(1 for t in test_results['tests'] if t['status'] == 'ERROR')
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Tests: {len(test_results['tests'])}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Duration: {total_duration:.2f} seconds")
    logger.info("-" * 50)
    
    # Print detailed results
    for test in test_results['tests']:
        status_emoji = '✅' if test['status'] == 'PASSED' else '❌'
        logger.info(f"{status_emoji} {test['name']:20} {test['status']:8} {test['duration']}")
        if test['error']:
            logger.error(f"   Error: {test['error']}")
    
    # Save detailed report
    report_file = os.path.join(TEST_DATA_DIR, 'test_report.json')
    try:
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"\nDetailed test report saved to: {report_file}")
    except Exception as e:
        logger.error(f"Failed to save test report: {str(e)}")
    
    # Return appropriate exit code (0 if all passed, 1 otherwise)
    return 0 if (failed == 0 and errors == 0) else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
