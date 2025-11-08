import os
import pytest
from unittest.mock import patch
from core.ingestion_service import IngestionService
from core.config_manager import ConfigManager

def reload_config_manager():
    """Forces a reload of the ConfigManager singleton for testing."""
    if hasattr(ConfigManager._instance, 'is_initialized'):
        del ConfigManager._instance.is_initialized
    return ConfigManager()

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Reloads config and clears env vars for each test."""
    # Clear env vars before the test
    vars_to_clear = ["CHUNK_SIZE", "CHUNK_OVERLAP"]
    original_values = {var: os.environ.get(var) for var in vars_to_clear}
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    # Force reload config to pick up cleared env vars
    reload_config_manager()

    yield

    # Restore original env vars
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]

    # Reload config again to restore original state
    reload_config_manager()

def test_chunk_size_override_from_env():
    """
    Tests if CHUNK_SIZE from env var overrides both basic and parent splitters.
    """
    test_chunk_size = "555"
    with patch.dict(os.environ, {"CHUNK_SIZE": test_chunk_size}):
        # Must reload config manager for it to pick up the patched env var
        reload_config_manager()

        # Re-instantiate IngestionService to trigger its __init__ with new config
        ingestion_service = IngestionService()

        # Assert that the main text_splitter uses the overridden value
        assert ingestion_service.text_splitter._chunk_size == int(test_chunk_size)

        # Assert that the parent_splitter also uses the overridden value
        assert ingestion_service.parent_splitter._chunk_size == int(test_chunk_size)

def test_chunk_size_falls_back_to_config():
    """
    Tests if IngestionService falls back to YAML config when env var is not set.
    """
    # No env var is set (handled by setup_and_teardown fixture)
    ingestion_service = IngestionService()

    # These values are from the default strategies.yaml
    expected_basic_chunk_size = 1000
    expected_parent_chunk_size = 2000

    assert ingestion_service.text_splitter._chunk_size == expected_basic_chunk_size
    assert ingestion_service.parent_splitter._chunk_size == expected_parent_chunk_size
