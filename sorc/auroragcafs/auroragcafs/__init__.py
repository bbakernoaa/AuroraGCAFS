"""
AuroraGCAFS - Adaptation of Microsoft's Aurora AI for NOAA's UFS/GEFS-aerosol output

This package provides tools for processing, analyzing, and visualizing
NOAA's UFS/GEFS-aerosol output using techniques inspired by Microsoft's Aurora AI.

The package consists of several main components:
- config: Configuration management
- data: Data loading and processing
- model: Deep learning models and training utilities
- utils: Utility functions for visualization and analysis
"""

import logging

__version__ = '0.1.0'

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import main components
from . import config
from . import data
from . import model
from . import utils

# Import commonly used classes and functions
from .config import config_manager
from .data import UFSAerosolLoader, GEFSAerosolLoader, DataProcessor
from .model import AerosolModel
from .utils import setup_logging, plot_map, plot_comparison

# Create default instances
ufs_loader = data.UFSAerosolLoader()
gefs_loader = data.GEFSAerosolLoader()
processor = data.DataProcessor()
model = model.AerosolModel()

# Set up default logging
utils.setup_logging()

__all__ = [
    # Modules
    'config', 'data', 'model', 'utils',

    # Classes
    'UFSAerosolLoader', 'GEFSAerosolLoader', 'DataProcessor', 'AerosolModel',

    # Functions
    'setup_logging', 'plot_map', 'plot_comparison',

    # Instances
    'config_manager', 'ufs_loader', 'gefs_loader', 'processor', 'model',
]