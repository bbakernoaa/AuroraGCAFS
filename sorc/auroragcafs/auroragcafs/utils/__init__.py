"""
Utility functions for AuroraGCAFS.

This module provides various utility functions for data handling,
visualization, and other common tasks.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from ..config import config_manager

# Set up module logger
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Parameters
    ----------
    level : str, optional
        Logging level. Default is "INFO".
    log_file : str, optional
        Path to log file. If None, logs to console only.
    """
    log_config = config_manager.get_logging_config()

    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_config.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or log_config.file)
        ]
    )
    logger.debug("Logging setup complete")

def plot_map(data: xr.DataArray,
            title: str = "",
            cmap: str = "viridis",
            projection: str = "PlateCarree",
            save_path: Optional[str] = None,
            **kwargs) -> plt.Figure:
    """
    Create a map plot of aerosol data.

    Parameters
    ----------
    data : xr.DataArray
        Data to plot
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap to use
    projection : str, optional
        Map projection to use
    save_path : str, optional
        Path to save the plot
    **kwargs : dict
        Additional arguments to pass to plotting function

    Returns
    -------
    plt.Figure
        The created figure

    Raises
    ------
    ValueError
        If projection is not supported
    """
    try:
        viz_config = config_manager.get_visualization_config()

        # Create figure and axes with specified projection
        proj = getattr(ccrs, projection)() if hasattr(ccrs, projection) else ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(viz_config.figure_width, viz_config.figure_height),
                              subplot_kw={'projection': proj})

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines()

        # Plot data
        im = data.plot(ax=ax, transform=ccrs.PlateCarree(),
                      cmap=cmap or viz_config.colormap,
                      **kwargs)

        # Set title
        if title:
            ax.set_title(title)

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=viz_config.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating map plot: {e}")
        raise

def plot_comparison(data1: xr.DataArray,
                   data2: xr.DataArray,
                   titles: Tuple[str, str] = ("Data 1", "Data 2"),
                   cmap: str = "viridis",
                   projection: str = "PlateCarree",
                   save_path: Optional[str] = None,
                   **kwargs) -> plt.Figure:
    """
    Create a side-by-side comparison plot.

    Parameters
    ----------
    data1 : xr.DataArray
        First dataset to plot
    data2 : xr.DataArray
        Second dataset to plot
    titles : tuple of str, optional
        Titles for each subplot
    cmap : str, optional
        Colormap to use
    projection : str, optional
        Map projection to use
    save_path : str, optional
        Path to save the plot
    **kwargs : dict
        Additional arguments to pass to plotting function

    Returns
    -------
    plt.Figure
        The created figure
    """
    try:
        viz_config = config_manager.get_visualization_config()

        # Create figure and axes
        proj = getattr(ccrs, projection)() if hasattr(ccrs, projection) else ccrs.PlateCarree()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(viz_config.figure_width * 2, viz_config.figure_height),
                                      subplot_kw={'projection': proj})

        # Add map features to both axes
        for ax in (ax1, ax2):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines()

        # Plot data
        im1 = data1.plot(ax=ax1, transform=ccrs.PlateCarree(),
                        cmap=cmap or viz_config.colormap,
                        **kwargs)
        im2 = data2.plot(ax=ax2, transform=ccrs.PlateCarree(),
                        cmap=cmap or viz_config.colormap,
                        **kwargs)

        # Set titles
        ax1.set_title(titles[0])
        ax2.set_title(titles[1])

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=viz_config.dpi, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        raise

def calculate_metrics(predictions: Union[np.ndarray, xr.DataArray],
                     targets: Union[np.ndarray, xr.DataArray]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    predictions : array-like
        Model predictions
    targets : array-like
        Target values

    Returns
    -------
    dict
        Dictionary containing various metrics
    """
    try:
        # Convert to numpy arrays if needed
        if isinstance(predictions, xr.DataArray):
            predictions = predictions.values
        if isinstance(targets, xr.DataArray):
            targets = targets.values

        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # Calculate correlation coefficient
        corr = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(corr)
        }

        logger.debug(f"Calculated metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def get_timestamp() -> str:
    """
    Get a formatted timestamp string.

    Returns
    -------
    str
        Current timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path

    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path

# Export functions
__all__ = ['setup_logging', 'plot_map', 'plot_comparison', 'calculate_metrics',
           'get_timestamp', 'ensure_directory']