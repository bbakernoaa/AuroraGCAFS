
import sys
import os
import time
import logging
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import xarray as xr
import numpy as np
import torch

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Mock the entire auroragcafs package since its internal structure is not correct
# This allows us to test the script's logic without fixing the package first.
MOCK_AURORAGCAFS = MagicMock()
MOCK_AURORAGCAFS.config_manager = MagicMock()
MOCK_AURORAGCAFS.ufs_loader = MagicMock()
MOCK_AURORAGCAFS.gefs_loader = MagicMock()
MOCK_AURORAGCAFS.processor = MagicMock()
MOCK_AURORAGCAFS.model.AerosolDataset = MagicMock()

sys.modules['auroragcafs'] = MOCK_AURORAGCAFS
sys.modules['auroragcafs.model'] = MOCK_AURORAGCAFS.model

from train_model import prepare_data

# Mock AerosolDataset since it's part of the auroragcafs package which is not installed
class MockAerosolDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data.time)

@pytest.fixture
def mock_config():
    """Fixture for a mock config object."""
    return MOCK_AURORAGCAFS.config_manager

def create_mock_dataset(date):
    """Creates a mock xarray Dataset for a given date."""
    time.sleep(0.1)  # Simulate a realistic I/O delay
    time_coord = pd.date_range(date, periods=1, freq='D')
    lat = np.arange(0, 10, 1)
    lon = np.arange(0, 10, 1)
    return xr.Dataset(
        {
            'aod': (('time', 'lat', 'lon'), np.random.rand(1, 10, 10)),
        },
        coords={'time': time_coord, 'lat': lat, 'lon': lon}
    )

def mock_load_and_process_day_success(date):
    """A pickleable mock function for _load_and_process_day."""
    return create_mock_dataset(date)

def mock_load_and_process_day_fail(date):
    """A pickleable mock function for _load_and_process_day that returns None."""
    return None

@patch('train_model.AerosolDataset', new=MockAerosolDataset)
def test_prepare_data_success(monkeypatch, mock_config):
    """Test prepare_data successfully loads, splits, and creates datasets."""
    # Arrange
    start_date = '20250101'
    end_date = '20250110'
    dates = pd.date_range(start_date, end_date, freq='D')

    monkeypatch.setattr('train_model._load_and_process_day', mock_load_and_process_day_success)

    # Act
    train_dataset, val_dataset = prepare_data(start_date, end_date, mock_config)

    # Assert
    total_samples = len(dates)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    assert isinstance(train_dataset, MockAerosolDataset)
    assert isinstance(val_dataset, MockAerosolDataset)
    assert len(train_dataset) == train_size
    assert len(val_dataset) == val_size

@patch('train_model.AerosolDataset', new=MockAerosolDataset)
def test_prepare_data_no_data(monkeypatch, mock_config):
    """Test prepare_data raises ValueError when no data is found."""
    # Arrange
    start_date = '20250101'
    end_date = '20250101'
    monkeypatch.setattr('train_model._load_and_process_day', mock_load_and_process_day_fail)

    # Act & Assert
    with pytest.raises(ValueError, match="No valid training data found"):
        prepare_data(start_date, end_date, mock_config)

def test_benchmark_prepare_data(monkeypatch, mock_config):
    """Benchmark the performance of the prepare_data function."""
    # Arrange
    start_date = '20250101'
    end_date = '20250110'
    monkeypatch.setattr('train_model._load_and_process_day', mock_load_and_process_day_success)

    # Act
    start_time = time.perf_counter()
    prepare_data(start_date, end_date, mock_config)
    end_time = time.perf_counter()

    # Assert
    duration = end_time - start_time
    logging.info(f"Benchmark prepare_data duration: {duration:.2f} seconds")
    # A reasonable expectation for 10 days with 0.1s delay each would be around 1 second,
    # but with multiprocessing it should be much faster.
    assert duration < 1.0
