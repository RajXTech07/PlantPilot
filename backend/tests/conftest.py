"""Pytest configuration and fixtures for PlantPilot tests."""

import pytest

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
