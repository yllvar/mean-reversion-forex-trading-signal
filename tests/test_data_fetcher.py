"""
Tests for the DataFetcher class.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metasync_dashboard.data.fetcher import DataFetcher
from metasync_dashboard.data.cache_manager import CacheManager
from metasync_dashboard.data.transformer import DataTransformer

# Sample test data
SAMPLE_OHLCV = {
    'timestamp': ['2023-01-01 00:00:00', '2023-01-01 00:01:00'],
    'open': [1.0, 1.1],
    'high': [1.2, 1.3],
    'low': [0.9, 1.0],
    'close': [1.1, 1.2],
    'volume': [1000, 2000]
}

SAMPLE_DF = pd.DataFrame(SAMPLE_OHLCV)
SAMPLE_DF['timestamp'] = pd.to_datetime(SAMPLE_DF['timestamp'])
SAMPLE_DF = SAMPLE_DF.set_index('timestamp')


@pytest.fixture
def mock_api_client():
    """Mock APIClient for testing."""
    with patch('metasync_dashboard.data.fetcher.APIClient') as mock:
        mock.return_value.get_ohlcv.return_value = SAMPLE_OHLCV
        yield mock


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager for testing."""
    with patch('metasync_dashboard.data.fetcher.CacheManager') as mock:
        instance = mock.return_value
        instance.is_cached.return_value = False
        instance.get_cached_data.return_value = None
        yield mock


@pytest.fixture
def mock_transformer():
    """Mock DataTransformer for testing."""
    with patch('metasync_dashboard.data.fetcher.DataTransformer') as mock:
        instance = mock.return_value
        instance.normalize_symbol.return_value = 'EUR/USD'
        instance.parse_ohlcv.return_value = SAMPLE_DF
        instance.add_technical_indicators.return_value = SAMPLE_DF
        instance.resample_data.return_value = SAMPLE_DF
        yield mock


@pytest.fixture
def data_fetcher(mock_api_client, mock_cache_manager, mock_transformer):
    """Fixture for DataFetcher with mocked dependencies."""
    return DataFetcher(api_key='test_key', cache_dir='/tmp/test_cache')


def test_init(data_fetcher, mock_api_client, mock_cache_manager, mock_transformer):
    """Test DataFetcher initialization."""
    assert data_fetcher is not None
    mock_api_client.assert_called_once_with(api_key='test_key')
    mock_cache_manager.assert_called_once_with(cache_dir='/tmp/test_cache')
    mock_transformer.assert_called_once()


def test_fetch_ohlcv_fresh(data_fetcher, mock_api_client):
    """Test fetching fresh OHLCV data."""
    # Setup
    symbol = 'EUR/USD'
    interval = '1min'
    exchange = 'OANDA'
    
    # Execute
    result = data_fetcher.fetch_ohlcv(symbol, interval, exchange)
    
    # Assert
    assert not result.empty
    assert len(result) == 2
    mock_api_client.return_value.get_ohlcv.assert_called_once_with(
        symbol=symbol,
        interval=interval,
        exchange=exchange
    )


def test_fetch_ohlcv_cached(data_fetcher, mock_cache_manager):
    """Test fetching OHLCV data from cache."""
    # Setup
    cache_mock = mock_cache_manager.return_value
    cache_mock.is_cached.return_value = True
    cache_mock.get_cached_data.return_value = SAMPLE_DF
    
    # Execute
    result = data_fetcher.fetch_ohlcv('EUR/USD', '1min', 'OANDA')
    
    # Assert
    assert not result.empty
    assert len(result) == 2
    cache_mock.get_cached_data.assert_called_once()


def test_fetch_ohlcv_api_error(data_fetcher, mock_api_client, mock_cache_manager):
    """Test API error handling with fallback to stale cache."""
    # Setup
    api_mock = mock_api_client.return_value
    api_mock.get_ohlcv.side_effect = Exception("API Error")
    
    cache_mock = mock_cache_manager.return_value
    cache_mock.is_cached.return_value = False  # No fresh cache
    cache_mock.get_cached_data.return_value = SAMPLE_DF  # But has stale cache
    
    # Execute - should use stale cache
    result = data_fetcher.fetch_ohlcv('EUR/USD', '1min', 'OANDA')
    
    # Assert
    assert not result.empty
    assert len(result) == 2
    cache_mock.get_cached_data.assert_called_once()


def test_fetch_multiple_symbols(data_fetcher):
    """Test fetching data for multiple symbols."""
    # Setup
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    
    # Execute
    results = data_fetcher.fetch_multiple_symbols(symbols)
    
    # Assert
    assert isinstance(results, dict)
    assert len(results) == 3
    for symbol in symbols:
        assert symbol in results
        assert not results[symbol].empty


def test_resample_data(data_fetcher, mock_transformer):
    """Test resampling data."""
    # Setup
    test_df = SAMPLE_DF.copy()
    interval = '5min'
    
    # Execute
    result = data_fetcher.resample_data(test_df, interval)
    
    # Assert
    assert not result.empty
    mock_transformer.return_value.resample_data.assert_called_once_with(
        test_df, interval
    )


def test_normalize_symbol(data_fetcher, mock_transformer):
    """Test symbol normalization."""
    # Setup
    test_symbol = 'EURUSD'
    expected = 'EUR/USD'
    
    # Execute
    data_fetcher.fetch_ohlcv(test_symbol, '1min', 'OANDA')
    
    # Assert
    mock_transformer.return_value.normalize_symbol.assert_called_once_with(test_symbol)


if __name__ == '__main__':
    pytest.main([__file__])
