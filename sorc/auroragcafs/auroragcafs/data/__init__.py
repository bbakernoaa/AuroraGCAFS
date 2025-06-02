"""
Data loading and processing module for AuroraGCAFS.

This module provides utilities for loading, processing, and transforming
NOAA's UFS/GEFS-aerosol output data for use in the AuroraGCAFS system.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import dask.array as da

from ..config import config_manager, DataConfig

# Set up module logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Base class for data loading operations.

    This class provides common functionality for loading and processing
    data from various sources.

    Parameters
    ----------
    config : DataConfig, optional
        Configuration for data loading operations. If not provided,
        will use default configuration.

    Attributes
    ----------
    config : DataConfig
        Data loading configuration
    cache_dir : Path
        Path to the cache directory
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the data loader."""
        self.config = config if config else config_manager.get_data_config()
        self.cache_dir = Path(os.path.expanduser(self.config.cache_dir))

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.debug(f"DataLoader initialized with cache directory: {self.cache_dir}")

    def get_cache_path(self, identifier: str) -> Path:
        """Get the path to a cached file."""
        return self.cache_dir / f"{identifier}.nc"

    def is_cached(self, identifier: str) -> bool:
        """Check if a file is cached."""
        return self.get_cache_path(identifier).exists()

    def _detect_file_format(self, file_path: str) -> str:
        """
        Detect the format of a data file.

        Parameters
        ----------
        file_path : str
            Path to the data file

        Returns
        -------
        str
            Format of the file ('netcdf' or 'grib')

        Raises
        ------
        ValueError
            If the file format cannot be determined
        """
        file_path = str(file_path)
        if file_path.endswith(('.nc', '.nc4', '.netcdf')):
            return 'netcdf'
        elif file_path.endswith(('.grib', '.grib2', '.grb', '.grb2')):
            return 'grib'
        else:
            # Try to detect by reading the file
            try:
                xr.open_dataset(file_path)
                return 'netcdf'
            except:
                try:
                    import grib2io
                    grib2io.open(file_path)
                    return 'grib'
                except:
                    raise ValueError(f"Could not determine format of file: {file_path}")

    def _open_dataset(self,
                     file_path: str,
                     format: Optional[str] = None,
                     parallel: bool = True,
                     **kwargs) -> xr.Dataset:
        """
        Open a dataset in either NetCDF or GRIB format.

        Parameters
        ----------
        file_path : str
            Path to the data file
        format : str, optional
            Format of the file ('netcdf' or 'grib'). If None, will detect.
        parallel : bool, optional
            Whether to use parallel processing. Default is True.
        **kwargs : dict
            Additional arguments to pass to the opener

        Returns
        -------
        xr.Dataset
            The opened dataset

        Raises
        ------
        ValueError
            If the file cannot be opened
        """
        if format is None:
            format = self._detect_file_format(file_path)

        try:
            if format == 'netcdf':
                return xr.open_dataset(file_path, **kwargs)
            elif format == 'grib':
                import grib2io
                # Use grib2io backend through xarray
                grb = grib2io.open(file_path)
                ds = xr.Dataset()

                # Load variables from GRIB file
                for var_name in grb.variables:
                    var = grb[var_name]
                    # Create DataArray with proper dimensions and coordinates
                    da = xr.DataArray(
                        data=var.values,
                        dims=['latitude', 'longitude'],
                        coords={
                            'latitude': var.lats,
                            'longitude': var.lons,
                            'time': var.validDate,
                            'level': var.level
                        },
                        attrs=var.attributes
                    )
                    ds[var_name] = da

                return ds
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            raise

class UFSAerosolLoader(DataLoader):
    """
    Loader for UFS aerosol data.

    This class handles loading and processing UFS aerosol data.

    Parameters
    ----------
    config : DataConfig, optional
        Configuration for data loading operations.

    Attributes
    ----------
    config : DataConfig
        Data loading configuration
    data_path : Path
        Path to the UFS data
    """

    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__(config)
        self.data_path = Path(os.path.expanduser(self.config.ufs_data_path))
        logger.info(f"UFSAerosolLoader initialized with data path: {self.data_path}")

    def load_forecast(self,
                     date: str,
                     forecast_hour: int,
                     variables: Optional[List[str]] = None,
                     use_cache: bool = True,
                     file_format: Optional[str] = None) -> xr.Dataset:
        """
        Load a UFS aerosol forecast.

        Parameters
        ----------
        date : str
            Forecast date in YYYYMMDD format
        forecast_hour : int
            Forecast hour
        variables : list of str, optional
            List of variables to load. If None, loads all available variables.
        use_cache : bool, optional
            Whether to use the cache. Default is True.
        file_format : str, optional
            Format of input files ('netcdf' or 'grib'). If None, will detect.

        Returns
        -------
        xr.Dataset
            Dataset containing the forecast data

        Raises
        ------
        FileNotFoundError
            If the forecast file is not found
        """
        cache_id = f"ufs_aerosol_{date}_f{forecast_hour:03d}"
        cache_path = self.get_cache_path(cache_id)

        if use_cache and cache_path.exists():
            logger.info(f"Loading UFS aerosol forecast from cache: {cache_path}")
            try:
                ds = xr.open_dataset(cache_path)
                if variables:
                    ds = ds[variables]
                return ds
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")

        # Look for both NetCDF and GRIB files if format not specified
        patterns = []
        if file_format in (None, 'netcdf'):
            patterns.append(f"{self.data_path}/ufs.{date}/*/aqm.t??z.aero_f{forecast_hour:03d}.nc")
        if file_format in (None, 'grib'):
            patterns.append(f"{self.data_path}/ufs.{date}/*/aqm.t??z.aero_f{forecast_hour:03d}.grb2")

        try:
            logger.info(f"Loading UFS aerosol forecast: {date}, forecast hour: {forecast_hour}")

            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    if len(files) > 1:
                        logger.warning(f"Multiple files found for pattern {pattern}, using first")

                    # Try to open the first matching file
                    try:
                        ds = self._open_dataset(files[0], format=file_format)
                        if variables:
                            ds = ds[variables]

                        if use_cache:
                            logger.debug(f"Caching UFS aerosol forecast to {cache_path}")
                            ds.to_netcdf(cache_path)

                        return ds
                    except Exception as e:
                        logger.warning(f"Failed to open {files[0]}: {e}")
                        continue

            raise FileNotFoundError(f"No valid files found matching patterns: {patterns}")

        except Exception as e:
            logger.error(f"Error loading UFS aerosol forecast: {e}")
            raise

class GEFSAerosolLoader(DataLoader):
    """
    Loader for GEFS aerosol data.

    This class handles loading and processing GEFS aerosol data.

    Parameters
    ----------
    config : DataConfig, optional
        Configuration for data loading operations.

    Attributes
    ----------
    config : DataConfig
        Data loading configuration
    data_path : Path
        Path to the GEFS data
    """

    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__(config)
        self.data_path = Path(os.path.expanduser(self.config.gefs_data_path))
        logger.info(f"GEFSAerosolLoader initialized with data path: {self.data_path}")

    def load_ensemble_forecast(self,
                             date: str,
                             forecast_hour: int,
                             ensemble_members: Optional[List[int]] = None,
                             variables: Optional[List[str]] = None,
                             use_cache: bool = True,
                             file_format: Optional[str] = None) -> xr.Dataset:
        """
        Load a GEFS aerosol ensemble forecast.

        Parameters
        ----------
        date : str
            Forecast date in YYYYMMDD format
        forecast_hour : int
            Forecast hour
        ensemble_members : list of int, optional
            List of ensemble member numbers to load. If None, loads all members.
        variables : list of str, optional
            List of variables to load. If None, loads all available variables.
        use_cache : bool, optional
            Whether to use the cache. Default is True.
        file_format : str, optional
            Format of input files ('netcdf' or 'grib'). If None, will detect.

        Returns
        -------
        xr.Dataset
            Dataset containing the ensemble forecast data

        Raises
        ------
        FileNotFoundError
            If the forecast files are not found
        """
        if ensemble_members is None:
            ensemble_members = list(range(1, 31))

        cache_id = f"gefs_aerosol_{date}_f{forecast_hour:03d}_e{len(ensemble_members)}"
        cache_path = self.get_cache_path(cache_id)

        if use_cache and cache_path.exists():
            logger.info(f"Loading GEFS ensemble forecast from cache: {cache_path}")
            try:
                ds = xr.open_dataset(cache_path)
                if variables:
                    ds = ds[variables]
                return ds
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")

        try:
            logger.info(f"Loading GEFS ensemble forecast: {date}, forecast hour: {forecast_hour}")
            datasets = []

            for member in ensemble_members:
                # Look for both NetCDF and GRIB files if format not specified
                patterns = []
                if file_format in (None, 'netcdf'):
                    patterns.append(f"{self.data_path}/gefs.{date}/gep{member:02d}/" +
                                 f"gefs.aero_f{forecast_hour:03d}.nc")
                if file_format in (None, 'grib'):
                    patterns.append(f"{self.data_path}/gefs.{date}/gep{member:02d}/" +
                                 f"gefs.aero_f{forecast_hour:03d}.grb2")

                ds_member = None
                for pattern in patterns:
                    files = glob.glob(pattern)
                    if files:
                        try:
                            ds_member = self._open_dataset(files[0], format=file_format)
                            break  # Successfully loaded a file
                        except Exception as e:
                            logger.warning(f"Failed to open {files[0]}: {e}")
                            continue

                if ds_member is None:
                    logger.warning(f"No valid files found for member {member}")
                    continue

                # Add ensemble member coordinate
                ds_member = ds_member.assign_coords(member=member)
                ds_member = ds_member.expand_dims("member")

                if variables:
                    ds_member = ds_member[variables]

                datasets.append(ds_member)

            if not datasets:
                raise FileNotFoundError("No valid files found for any ensemble members")

            ds = xr.concat(datasets, dim="member")

            if use_cache:
                logger.debug(f"Caching GEFS ensemble forecast to {cache_path}")
                ds.to_netcdf(cache_path)

            return ds

        except Exception as e:
            logger.error(f"Error loading GEFS ensemble forecast: {e}")
            raise

class DataProcessor:
    """
    Data processing operations for aerosol data.

    Parameters
    ----------
    config : dict, optional
        Configuration for data processing operations.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config if config else config_manager.get_processing_config()
        logger.debug(f"DataProcessor initialized with config: {self.config}")

    def normalize(self, data: Union[xr.DataArray, xr.Dataset],
                 method: Optional[str] = None,
                 per_variable: Optional[bool] = None,
                 scale_factors: Optional[Dict[str, float]] = None,
                 bounds: Optional[Dict[str, float]] = None,
                 epsilon: Optional[float] = None,
                 pre_log_transform: Optional[Dict] = None) -> Union[xr.DataArray, xr.Dataset]:
        """
        Normalize data using various methods with support for per-variable configurations.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            Data to normalize
        method : str, optional
            Normalization method: "min_max", "z_score", "log", or "none".
            If None, uses the method from config.
        per_variable : bool, optional
            Whether to normalize each variable separately.
            If None, uses the setting from config.
        scale_factors : dict, optional
            Scale factors per variable.
            If None, uses the factors from config.
        bounds : dict, optional
            Bounds for min_max normalization.
            If None, uses the bounds from config.
        epsilon : float, optional
            Small constant to avoid division by zero.
            If None, uses the value from config.
        pre_log_transform : dict, optional
            Settings for log transformation before normalization.
            If None, uses the settings from config.

        Returns
        -------
        xr.DataArray or xr.Dataset
            Normalized data

        Raises
        ------
        ValueError
            If method is not supported or if required parameters are missing
        """
        # Get normalization settings from config if not provided
        norm_config = self.config.normalization
        method = method or norm_config.method
        per_variable = per_variable if per_variable is not None else norm_config.per_variable
        scale_factors = scale_factors or norm_config.scale_factors
        bounds = bounds or norm_config.bounds
        epsilon = epsilon if epsilon is not None else norm_config.epsilon
        pre_log_transform = pre_log_transform or getattr(norm_config, 'pre_log_transform', {})

        logger.debug(f"Normalizing data using method: {method}, per_variable: {per_variable}")

        def _apply_log_transform(arr: xr.DataArray, var_name: Optional[str] = None) -> xr.DataArray:
            """Apply log transformation with configurable base and offset"""
            if not pre_log_transform.get('enabled', False):
                return arr

            # Check if this variable should be log-transformed
            variables = pre_log_transform.get('variables', [])
            if variables and var_name not in variables:
                return arr

            # Get log settings
            base = pre_log_transform.get('base', 'e')
            offset = float(pre_log_transform.get('offset', 1.0))

            # Apply log transform
            if base == 'e':
                return xr.apply_ufunc(np.log1p, arr + (offset - 1))
            elif base == '10':
                return xr.apply_ufunc(np.log10, arr + offset)
            else:
                raise ValueError(f"Unsupported log base: {base}")

        def _normalize_array(arr: xr.DataArray, var_name: Optional[str] = None) -> xr.DataArray:
            """Helper function to normalize a single DataArray"""
            # Apply pre-log transform if configured
            arr = _apply_log_transform(arr, var_name)

            # Apply scale factor if available
            if var_name and var_name in scale_factors:
                arr = arr * scale_factors[var_name]

            if method.lower() == "min_max":
                data_min = float(arr.min())
                data_max = float(arr.max())
                # Use configured bounds if provided
                output_min = bounds.get('min', 0.0)
                output_max = bounds.get('max', 1.0)

                # Handle constant values
                if abs(data_max - data_min) < epsilon:
                    if data_min < output_min:
                        normalized = xr.full_like(arr, output_min)
                    elif data_min > output_max:
                        normalized = xr.full_like(arr, output_max)
                    else:
                        normalized = xr.full_like(arr, data_min)
                else:
                    normalized = (arr - data_min) / (data_max - data_min + epsilon)
                    # Scale to desired range
                    normalized = normalized * (output_max - output_min) + output_min

            elif method.lower() == "z_score":
                mean = float(arr.mean())
                std = float(arr.std())
                if std < epsilon:
                    normalized = xr.zeros_like(arr)
                else:
                    normalized = (arr - mean) / (std + epsilon)

            elif method.lower() == "log":
                # This is different from pre_log_transform as it's a normalization method
                normalized = xr.apply_ufunc(np.log1p, arr + epsilon)

            elif method.lower() == "none":
                normalized = arr

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            logger.debug(f"Normalized variable {var_name if var_name else 'data'} "
                        f"using method {method}")
            return normalized

        # Handle Dataset case
        if isinstance(data, xr.Dataset):
            if per_variable:
                return xr.Dataset({
                    var: _normalize_array(data[var], var)
                    for var in data.data_vars
                })
            else:
                # Convert to array, normalize, and convert back
                try:
                    stacked = xr.concat([data[var] for var in data.data_vars],
                                      dim='variable')
                    normalized = _normalize_array(stacked)
                    # Unstack and create new dataset
                    return xr.Dataset({
                        var: normalized.sel(variable=i)
                        for i, var in enumerate(data.data_vars)
                    })
                except Exception as e:
                    logger.error(f"Error during joint normalization: {e}")
                    raise

        # Handle DataArray case
        return _normalize_array(data)

    def regrid(self, data: xr.DataArray,
               target_resolution: float,
               method: str = "bilinear") -> xr.DataArray:
        """
        Regrid data to a different resolution.

        Parameters
        ----------
        data : xr.DataArray
            Data to regrid
        target_resolution : float
            Target grid resolution in degrees
        method : str, optional
            Interpolation method: "nearest", "bilinear", or "conservative"

        Returns
        -------
        xr.DataArray
            Regridded data
        """
        logger.info(f"Regridding data to {target_resolution}° resolution")

        # Create target grid
        lat = np.arange(-90, 90 + target_resolution, target_resolution)
        lon = np.arange(-180, 180 + target_resolution, target_resolution)

        return data.interp(lat=lat, lon=lon, method=method)

    def calculate_ensemble_stats(self, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
        """
        Calculate ensemble statistics.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing ensemble members

        Returns
        -------
        dict
            Dictionary containing ensemble mean and spread
        """
        logger.debug("Calculating ensemble statistics")

        stats = {
            'mean': ds.mean(dim='member'),
            'std': ds.std(dim='member'),
            'min': ds.min(dim='member'),
            'max': ds.max(dim='member'),
            'median': ds.median(dim='member')
        }

        return stats

# Create default instances
ufs_loader = UFSAerosolLoader()
gefs_loader = GEFSAerosolLoader()
processor = DataProcessor()

__all__ = ['DataLoader', 'UFSAerosolLoader', 'GEFSAerosolLoader', 'DataProcessor',
           'ufs_loader', 'gefs_loader', 'processor']