"""Tests for configuration management."""
import os
from pathlib import Path

import pytest

from metasync_dashboard.config import settings

def test_settings_initialization():
    """Test that settings are properly initialized."""
    assert settings.API_KEY == "test_api_key"
    assert settings.API_BASE_URL == "https://api.twelvedata.com"
    assert settings.LOG_LEVEL == "DEBUG"

def test_directories_created():
    """Test that required directories are created."""
    assert settings.DATA_DIR.exists()
    assert settings.CACHE_DIR.exists()
    assert settings.LOGS_DIR.exists()

def test_environment_variables(monkeypatch):
    """Test that environment variables override defaults."
    monkeypatch.setenv("API_BASE_URL", "https://test-api.example.com")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    
    from metasync_dashboard.config import settings
    
    assert settings.API_BASE_URL == "https://test-api.example.com"
    assert settings.LOG_LEVEL == "INFO"
