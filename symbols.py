"""
Symbol handling utilities for the trading system.
Provides consistent symbol formatting and validation.
"""
import re
from typing import List, Optional, Tuple, Set

# Default symbols for the trading system
DEFAULT_SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD"]

class SymbolFormatError(ValueError):
    """Raised when a symbol has an invalid format."""
    pass

def normalize_symbol(symbol: str) -> str:
    """
    Normalize a trading symbol to a standard format (e.g., 'eurusd' -> 'EUR/USD').
    
    Args:
        symbol: The symbol to normalize (e.g., 'eurusd', 'EURUSD', 'EUR/USD')
        
    Returns:
        str: Normalized symbol in 'XXX/YYY' format (e.g., 'EUR/USD')
        
    Raises:
        SymbolFormatError: If the symbol cannot be normalized
    """
    if not symbol or not isinstance(symbol, str):
        raise SymbolFormatError(f"Invalid symbol: {symbol}")
        
    # Remove any whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Handle common formats:
    # 1. Already in 'XXX/YYY' format
    if '/' in symbol:
        base, quote = symbol.split('/')
    # 2. No separator (e.g., 'EURUSD')
    elif len(symbol) == 6:
        base, quote = symbol[:3], symbol[3:]
    # 3. Other formats (e.g., 'BTC-USD')
    elif '-' in symbol:
        base, quote = symbol.split('-')
    else:
        # Try to split into 3-character pairs if possible
        if len(symbol) >= 6:
            base, quote = symbol[:3], symbol[3:6]
        else:
            raise SymbolFormatError(f"Cannot parse symbol format: {symbol}")
    
    # Validate base and quote currencies
    if not (len(base) == 3 and base.isalpha() and len(quote) == 3 and quote.isalpha()):
        raise SymbolFormatError(f"Invalid currency pair format: {base}/{quote}")
    
    return f"{base}/{quote}"

def symbol_to_filename(symbol: str) -> str:
    """
    Convert a symbol to a filesystem-safe string.
    
    Args:
        symbol: The symbol to convert (e.g., 'EUR/USD')
        
    Returns:
        str: Filesystem-safe string (e.g., 'EUR_USD')
    """
    normalized = normalize_symbol(symbol)
    return normalized.replace('/', '_')

def is_valid_symbol(symbol: str) -> bool:
    """
    Check if a symbol is in a valid format.
    
    Args:
        symbol: The symbol to validate
        
    Returns:
        bool: True if the symbol is valid, False otherwise
    """
    try:
        normalize_symbol(symbol)
        return True
    except SymbolFormatError:
        return False

def get_quote_currency(symbol: str) -> str:
    """
    Get the quote currency from a symbol.
    
    Args:
        symbol: The symbol (e.g., 'EUR/USD')
        
    Returns:
        str: The quote currency (e.g., 'USD')
    """
    return normalize_symbol(symbol).split('/')[1]

def get_base_currency(symbol: str) -> str:
    """
    Get the base currency from a symbol.
    
    Args:
        symbol: The symbol (e.g., 'EUR/USD')
        
    Returns:
        str: The base currency (e.g., 'EUR')
    """
    return normalize_symbol(symbol).split('/')[0]

def get_symbols() -> List[str]:
    """
    Get the list of supported trading symbols.
    
    Returns:
        List[str]: List of supported symbols in 'XXX/YYY' format
    """
    return DEFAULT_SYMBOLS.copy()

def validate_symbols(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of symbols.
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        Tuple containing:
            - List of valid symbols
            - List of invalid symbols
    """
    valid = []
    invalid = []
    
    for symbol in symbols:
        try:
            valid.append(normalize_symbol(symbol))
        except SymbolFormatError:
            invalid.append(symbol)
            
    return valid, invalid

# Example usage
if __name__ == "__main__":
    # Test normalization
    test_symbols = ["EURUSD", "GBP/USD", "usdjpy", "XAU-USD", "BTC/USD"]
    
    print("Symbol normalization:")
    for symbol in test_symbols:
        try:
            normalized = normalize_symbol(symbol)
            print(f"{symbol} -> {normalized}")
        except SymbolFormatError as e:
            print(f"Error with {symbol}: {e}")
    
    # Test symbol validation
    print("\nSymbol validation:")
    valid, invalid = validate_symbols(test_symbols)
    print(f"Valid: {valid}")
    print(f"Invalid: {invalid}")
