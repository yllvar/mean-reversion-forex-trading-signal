"""
API client for fetching market data from various data providers.
"""
import logging
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter, Retry

from .. import config

logger = logging.getLogger(__name__)

class APIClient:
    """Client for making API requests with retries and rate limiting."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
        """
        self.base_url = base_url or config.BASE_URL
        self.api_key = api_key or config.API_KEY
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Mount the retry strategy to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the API.
        
        Args:
            endpoint: API endpoint (e.g., 'time_series')
            params: Query parameters
            
        Returns:
            JSON response from the API
            
        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
        """
        if params is None:
            params = {}
            
        # Add API key to the request parameters
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=settings.API_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_ohlcv(
        self, 
        symbol: str, 
        interval: str = "1min", 
        exchange: str = "OANDA",
        output_size: int = 1000
    ) -> Dict[str, Any]:
        """Get OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'EUR/USD')
            interval: Data interval (e.g., '1min', '5min', '15min')
            exchange: Exchange to get data from
            output_size: Number of data points to return
            
        Returns:
            Dictionary containing OHLCV data
        """
        # Format symbol for the API (replace / with %2F)
        formatted_symbol = symbol.replace("/", "%2F")
        
        params = {
            "symbol": formatted_symbol,
            "interval": interval,
            "exchange": exchange,
            "outputsize": output_size,
            "format": "JSON",
            "type": "stock",
            "timezone": "UTC"
        }
        
        return self.get("time_series", params=params)
