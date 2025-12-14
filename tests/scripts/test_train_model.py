"""
Unit tests for the AuroraGCAFS training script.
"""
import sys
from unittest.mock import MagicMock

# Mock heavy/problematic dependencies before they are imported by the application code.
# This prevents ImportError when running tests in an environment where not all
# heavy dependencies (like grib2io) are installed.
sys.modules['grib2io'] = MagicMock()
sys.modules['netCDF4'] = MagicMock()
sys.modules['cfgrib'] = MagicMock()
sys.modules['cartopy'] = MagicMock()

import pytest
import pandas as pd
import xarray as xr

import os

# Add the 'scripts' directory to the system path to allow for robust imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

# Now that the path is configured, we can import the module
from scripts import train_model

def test_load_and_process_day_success(mocker):
    """
    Tests the successful loading and processing of a single day's data.
    """
    # Arrange
    mock_date = pd.Timestamp('2025-01-01')

    # Mock the ufs_loader and processor modules
    mock_ds = xr.Dataset({'a': (('x', 'y'), [[1, 2], [3, 4]])})
    mocker.patch('scripts.train_model.ufs_loader.load_forecast', return_value=mock_ds)
    mocker.patch('scripts.train_model.processor.normalize', return_value=mock_ds)

    # Act
    result = _load_and_process_day(mock_date)

    # Assert
    assert result is not None
    assert isinstance(result, xr.Dataset)
    scripts.train_model.ufs_loader.load_forecast.assert_called_once_with('20250101', 0)
    scripts.train_model.processor.normalize.assert_called_once_with(mock_ds)

def test_load_and_process_day_failure(mocker):
    """
    Tests the failure scenario where the ufs_loader raises an exception.
    """
    # Arrange
    mock_date = pd.Timestamp('2025-01-01')

    # Mock the ufs_loader to raise an exception
    mocker.patch('scripts.train_model.ufs_loader.load_forecast', side_effect=IOError("File not found"))

    # Act
    result = _load_and_process_day(mock_date)

    # Assert
    assert result is None
    scripts.train_model.ufs_loader.load_forecast.assert_called_once_with('20250101', 0)
