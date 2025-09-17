"""
Test script to verify the signal processing pipeline.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_signal_processing():
    """Test the signal processing pipeline with sample data."""
    from signals import MeanReversionStrategy, process_signals
    
    logger.info("\n=== Testing Signal Processing Pipeline ===")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq='1min')
    data = {
        'open': 1.08 + np.random.randn(100) * 0.001,
        'high': 1.081 + np.random.rand(100) * 0.002,
        'low': 1.079 - np.random.rand(100) * 0.002,
        'close': 1.08 + np.random.randn(100) * 0.001,
        'volume': 1000 + np.random.randint(-100, 100, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Test with a single symbol
    test_data = {'EUR/USD': df}
    
    # Process signals
    logger.info("Processing signals...")
    signals = process_signals(test_data)
    
    # Verify output format
    assert isinstance(signals, dict), "Signals should be a dictionary"
    assert 'EUR/USD' in signals, "Should have signal for EUR/USD"
    
    signal = signals['EUR/USD']
    assert isinstance(signal, tuple), "Signal should be a tuple"
    assert len(signal) == 3, "Signal tuple should have 3 elements (signal, price, z_score)"
    assert signal[0] in ['BUY', 'SELL', 'HOLD'], f"Invalid signal type: {signal[0]}"
    assert isinstance(signal[1], (int, float)), "Price should be a number"
    assert isinstance(signal[2], (int, float)), "Z-score should be a number"
    
    logger.info("\nSignal processing test passed!")
    logger.info(f"Signal for EUR/USD: {signal}")
    
    # Test with empty data
    logger.info("\nTesting with empty data...")
    empty_signals = process_signals({})
    assert empty_signals == {}, "Should return empty dict for empty input"
    
    logger.info("Empty data test passed!")
    
    return True

def test_data_fetcher():
    """Test the data fetcher with a single symbol."""
    from data_fetcher import fetch_ohlcv
    
    logger.info("\n=== Testing Data Fetcher ===")
    
    # Test with a single symbol
    symbol = 'EUR/USD'
    logger.info(f"Fetching data for {symbol}...")
    df = fetch_ohlcv(symbol, output_size=5)  # Only fetch 5 data points
    
    assert df is not None, "DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 0, "Should have at least one data point"
    
    # Check for required OHLC columns
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
        
    # Volume is optional but recommended
    if 'volume' not in df.columns:
        logger.warning("Volume data not available in the response")
    
    logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
    logger.info(f"Data sample:\n{df.head()}")
    
    return True

def run_tests():
    """Run all tests and report results."""
    tests = [
        ("Signal Processing", test_signal_processing),
        ("Data Fetcher", test_data_fetcher)
    ]
    
    print("\n" + "="*50)
    print("Running Tests")
    print("="*50)
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            print("-" * (len(name) + 1))
            success = test_func()
            status = "PASSED" if success else "FAILED"
            results.append((name, status, ""))
            print(f"\n✅ {name}: {status}")
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"\n❌ {name} failed: {error_msg}")
            print("\nTraceback:")
            traceback.print_exc()
            results.append((name, "ERROR", error_msg))
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    for name, status, error in results:
        print(f"{name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    # Return non-zero exit code if any test failed
    if any(status != "PASSED" for _, status, _ in results):
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
