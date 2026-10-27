import os
import pytest

# Ensure deterministic and mocked tests run without requiring a live NVIDIA API key
os.environ.setdefault("NVIDIA_API_KEY", "mock-test-key-for-deterministic-tests")

from plantpilot.config import reset_config
from plantpilot.storage import reset_database
from plantpilot.digital_twin import reset_twin_store
from plantpilot.workforce import reset_workforce_store


@pytest.fixture(autouse=True)
def reset_state():
    """Reset configuration, database, twin store, and workforce state before each test."""
    reset_config()
    reset_database()
    reset_twin_store()
    reset_workforce_store()
    yield
    reset_config()
    reset_database()
    reset_twin_store()
    reset_workforce_store()
