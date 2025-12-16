
import sys
import pytest
from unittest.mock import MagicMock
import time
import psutil
import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# A mock class to replace the real AerosolDataset during the test
class MockAerosolDataset:
    def __init__(self, data):
        self.data = data

@pytest.fixture
def synthetic_data_factory(tmp_path):
    """Factory to create synthetic NetCDF data for testing."""
    def _create_data(num_days=30):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        date_range = pd.to_datetime(np.arange(num_days), unit='D', origin='2025-01-01')

        for date in date_range:
            day_str = date.strftime('%Y%m%d')
            forecast_hour = 0
            cache_id = f"ufs_aerosol_{day_str}_f{forecast_hour:03d}"
            file_path = cache_dir / f"{cache_id}.nc"

            ds = xr.Dataset(
                {
                    'aerosol_mass': (('time', 'lat', 'lon'), np.random.rand(1, 181, 360)),
                },
                coords={
                    'time': [date],
                    'lat': np.linspace(-90, 90, 181),
                    'lon': np.linspace(-180, 180, 360)
                }
            )
            ds.to_netcdf(file_path)
        return str(cache_dir)
    return _create_data


def track_memory_usage(func, *args, **kwargs):
    """
    Tracks memory usage of a function call.
    Returns the peak memory usage in MiB and the function's return value.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    result = func(*args, **kwargs)
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_mem = mem_after - mem_before
    return peak_mem, result

def test_prepare_data_performance(benchmark, synthetic_data_factory, monkeypatch):
    """
    Benchmarks the memory and time performance of the prepare_data function.
    """
    # 1. Mock the entire auroragcafs package before it's imported by train_model
    #    to prevent import-time side effects (like the ConfigManager error).
    mock_ufs_loader = MagicMock()
    mock_processor = MagicMock()
    mock_config_manager = MagicMock()

    mock_auroragcafs = MagicMock()
    mock_auroragcafs.ufs_loader = mock_ufs_loader
    mock_auroragcafs.processor = mock_processor
    mock_auroragcafs.config_manager = mock_config_manager

    monkeypatch.setitem(sys.modules, 'auroragcafs', mock_auroragcafs)
    # Also mock the model submodule which is imported by train_model
    monkeypatch.setitem(sys.modules, 'auroragcafs.model', MagicMock())

    # 2. Add the 'scripts' directory to the path and import the module under test.
    #    This is done inside the test to ensure mocks are active first.
    scripts_dir = Path(__file__).parent.parent / 'scripts'
    monkeypatch.syspath_prepend(str(scripts_dir))
    import train_model

    # 3. Configure the mocks and test data
    num_days = 10
    cache_dir = synthetic_data_factory(num_days=num_days)

    mock_ufs_loader.get_cache_path.side_effect = lambda cache_id: Path(cache_dir) / f"{cache_id}.nc"
    mock_ufs_loader.load_forecast.return_value = None  # This is only called for caching
    mock_processor.normalize = lambda x: x  # Return data as is

    # Replace the real AerosolDataset with our mock
    monkeypatch.setattr(train_model, 'AerosolDataset', MockAerosolDataset)

    # 4. Define the function to be benchmarked
    def run_prepare_data():
        train_model.prepare_data(
            start_date='20250101',
            end_date=f'202501{num_days}',
        )

    # 5. Run the benchmarks
    # Benchmark execution time
    benchmark(run_prepare_data)

    # Measure memory usage
    peak_mem, _ = track_memory_usage(run_prepare_data)
    print(f"Peak memory usage for prepare_data: {peak_mem:.2f} MiB")

    # Add memory usage to benchmark results for reporting
    benchmark.extra_info['peak_memory_mib'] = peak_mem

if __name__ == "__main__":
    pytest.main(['-v', __file__])
