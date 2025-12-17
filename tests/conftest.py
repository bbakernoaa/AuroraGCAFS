
import sys
from unittest.mock import MagicMock
import pytest

@pytest.fixture(autouse=True)
def mock_auroragcafs_package(monkeypatch):
    """
    Mocks the entire auroragcafs package before any tests are imported.
    This prevents ImportError due to the package's structure.
    """
    mock_auroragcafs = MagicMock()
    sys.modules['auroragcafs'] = mock_auroragcafs
    sys.modules['auroragcafs.model'] = MagicMock()
