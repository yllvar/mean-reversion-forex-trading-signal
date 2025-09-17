"""Pytest configuration and fixtures."""
import os
import shutil
from pathlib import Path

import pytest

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir

@pytest.fixture(scope="function")
def temp_cache_dir():
    """Fixture for temporary cache directory."""
    cache_dir = Path("tests/temp_cache")
    cache_dir.mkdir(exist_ok=True)
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
