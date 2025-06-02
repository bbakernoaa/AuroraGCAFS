"""
Configuration module for AuroraGCAFS.

This module provides configuration management for the AuroraGCAFS package,
handling loading, validation, and access to configuration settings.
"""

import os
import logging
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

# Setup logger
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))), "parm", "config.yaml")

@dataclass
class AerosolConfig:
    """
    Configuration for aerosol data processing

    Parameters
    ----------
    species : List[str]
        List of aerosol species names to process
    vertical_levels : List[int]
        List of vertical levels to include
    spatial_resolution : float
        Spatial resolution in degrees
    time_step : int
        Time step in hours
    """
    species: List[str]
    vertical_levels: List[int]
    spatial_resolution: float
    time_step: int  # in hours

@dataclass
class ModelConfig:
    """
    Configuration for machine learning models

    Parameters
    ----------
    model_type : str
        Type of model to use (e.g., unet, resnet)
    input_features : List[str]
        List of input features
    hidden_layers : List[int]
        List of neurons in each hidden layer
    activation : str
        Activation function
    learning_rate : float
        Learning rate for optimization
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    optimizer : str
        Optimizer to use
    loss_function : str
        Loss function to use
    use_gpu : bool
        Whether to use GPU acceleration
    """
    model_type: str
    input_features: List[str]
    hidden_layers: List[int]
    activation: str
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"
    loss_function: str = "mse"
    use_gpu: bool = True

@dataclass
class DataConfig:
    """
    Configuration for data sources

    Parameters
    ----------
    ufs_data_path : str
        Path to UFS aerosol data
    gefs_data_path : str
        Path to GEFS-aerosol data
    output_path : str
        Path for output data
    cache_dir : str
        Directory for caching intermediate results
    historical_data : Optional[str]
        Path to historical data, if available
    """
    ufs_data_path: str
    gefs_data_path: str
    output_path: str
    cache_dir: str
    historical_data: Optional[str] = None

@dataclass
class VisualizationConfig:
    """
    Configuration for visualization

    Parameters
    ----------
    map_projection : str
        Cartopy map projection to use
    colormap : str
        Matplotlib colormap
    dpi : int
        DPI for image output
    figure_width : int
        Width of figure in inches
    figure_height : int
        Height of figure in inches
    """
    map_projection: str
    colormap: str
    dpi: int
    figure_width: int = 10
    figure_height: int = 8

@dataclass
class LoggingConfig:
    """
    Configuration for logging

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    format : str
        Log message format string
    file : str
        Path to log file
    """
    level: str
    format: str
    file: str

@dataclass
class SystemConfig:
    """
    Configuration for system settings

    Parameters
    ----------
    n_workers : int
        Number of worker processes
    memory_limit : str
        Memory limit per worker
    enable_dask : bool
        Whether to enable Dask for parallel processing
    """
    n_workers: int
    memory_limit: str
    enable_dask: bool

@dataclass
class PreLogTransformConfig:
    """
    Configuration for pre-log transformation

    Parameters
    ----------
    enabled : bool
        Whether to apply log transform before normalization
    variables : List[str]
        List of variables to apply log transform to
    base : str
        Log base to use ('e' or '10')
    offset : float
        Added before taking log to handle zeros: log(x + offset)
    """
    enabled: bool = False
    variables: List[str] = None
    base: str = "e"
    offset: float = 1.0

    def __post_init__(self):
        """Validate and set default values"""
        if self.variables is None:
            self.variables = []
        if self.base not in ["e", "10"]:
            raise ValueError(f"Unsupported log base: {self.base}")

@dataclass
class NormalizationConfig:
    """
    Configuration for data normalization

    Parameters
    ----------
    method : str
        Normalization method to use: "min_max", "z_score", "log", or "none"
    per_variable : bool
        Whether to normalize each variable separately
    scale_factors : Dict[str, float]
        Scale factors to apply per variable
    bounds : Dict[str, float]
        Bounds for min_max normalization (min and max values)
    epsilon : float
        Small constant to avoid division by zero
    pre_log_transform : PreLogTransformConfig
        Configuration for pre-log transformation
    """
    method: str = "min_max"
    per_variable: bool = True
    scale_factors: Dict[str, float] = None
    bounds: Dict[str, float] = None
    epsilon: float = 1e-8
    pre_log_transform: PreLogTransformConfig = None

    def __post_init__(self):
        """Validate and set default values"""
        if self.method not in ["min_max", "z_score", "log", "none"]:
            raise ValueError(f"Invalid normalization method: {self.method}")

        if self.scale_factors is None:
            self.scale_factors = {}

        if self.bounds is None:
            self.bounds = {"min": 0.0, "max": 1.0}

        if "min" not in self.bounds or "max" not in self.bounds:
            raise ValueError("Bounds must contain 'min' and 'max' keys")

        if self.pre_log_transform is None:
            self.pre_log_transform = PreLogTransformConfig()

@dataclass
class ProcessingConfig:
    """
    Configuration for data processing

    Parameters
    ----------
    variable_names : List[str]
        Names of variables to process
    resolution : float
        Spatial resolution in degrees
    temporal_frequency : str
        Temporal frequency of data
    interpolation_method : str
        Method for interpolation
    normalization : NormalizationConfig
        Configuration for data normalization
    """
    variable_names: List[str]
    resolution: float
    temporal_frequency: str
    interpolation_method: str = "bilinear"
    normalization: NormalizationConfig = None

    def __post_init__(self):
        """Set default normalization config if none provided"""
        if self.normalization is None:
            self.normalization = NormalizationConfig()

class ConfigManager:
    """
    Manager for AuroraGCAFS configuration.

    This class manages configuration for the AuroraGCAFS package,
    handling loading from files, validation, and providing structured access.

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file. Default is CONFIG_PATH.

    Attributes
    ----------
    config_file : str
        Path to the configuration file
    config : dict
        Configuration dictionary
    """

    def __init__(self, config_file: str = CONFIG_PATH):
        """
        Initialize the configuration manager.

        Parameters
        ----------
        config_file : str, optional
            Path to the configuration file. Default is CONFIG_PATH.
        """
        self.config_file = config_file
        self.config = {}

        if os.path.exists(config_file):
            logger.info(f"Loading configuration from {config_file}")
            self._load_yaml_config()
        else:
            logger.warning(f"Configuration file {config_file} not found. Creating default configuration.")
            self._create_default_config()

    def _load_yaml_config(self):
        """
        Load configuration from YAML file.

        Raises
        ------
        yaml.YAMLError
            If YAML parsing fails
        """
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Successfully loaded YAML config from {self.config_file}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def _create_default_config(self):
        """
        Create default configuration if none exists.

        Creates a default configuration dictionary with sensible defaults for
        aerosol processing, model parameters, and data paths, then saves to file.
        """
        self.config = {
            "data": {
                "ufs_data_path": "./data/ufs",
                "gefs_data_path": "./data/gefs",
                "output_path": "./output",
                "cache_dir": "~/.auroragcafs/cache",
            },
            "model": {
                "model_type": "unet",
                "input_features": ["aod", "pm25", "pm10"],
                "hidden_layers": [64, 128, 256, 128, 64],
                "activation": "relu",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "optimizer": "adam",
                "loss_function": "mse",
                "use_gpu": True,
            },
            "processing": {
                "variable_names": ["dust", "sea_salt", "sulfate", "organic_carbon", "black_carbon"],
                "vertical_levels": [1, 2, 3, 5, 7, 10],
                "resolution": 0.25,
                "temporal_frequency": "6h",
                "normalization": "min_max",
                "interpolation_method": "bilinear",
            },
            "visualization": {
                "map_projection": "PlateCarree",
                "colormap": "viridis",
                "dpi": 300,
                "figure_width": 10,
                "figure_height": 8,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/auroragcafs.log",
            },
            "system": {
                "n_workers": 4,
                "memory_limit": "8GB",
                "enable_dask": True,
            }
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Created default YAML configuration at {self.config_file}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Parameters
        ----------
        section : str
            The configuration section
        key : str
            The configuration key within the section
        default : Any, optional
            Default value to return if the key is not found

        Returns
        -------
        Any
            The configuration value
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except (KeyError, AttributeError) as e:
            logger.warning(f"Configuration key {section}.{key} not found: {e}")
            return default

    def get_aerosol_config(self) -> AerosolConfig:
        """
        Get aerosol configuration.

        Returns
        -------
        AerosolConfig
            Configuration dataclass for aerosol data processing

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            processing = self.config.get("processing", {})
            config = AerosolConfig(
                species=processing.get("variable_names", []),
                vertical_levels=processing.get("vertical_levels", []),
                spatial_resolution=float(processing.get("resolution", 0.25)),
                time_step=int(processing.get("temporal_frequency", "6h").replace("h", ""))
            )
            logger.debug(f"Loaded aerosol config: {config}")
            return config
        except (KeyError, ValueError) as e:
            logger.error(f"Error loading aerosol configuration: {e}")
            raise

    def get_model_config(self) -> ModelConfig:
        """
        Get model configuration.

        Returns
        -------
        ModelConfig
            Configuration dataclass for machine learning models

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            model = self.config.get("model", {})
            config = ModelConfig(
                model_type=model.get("model_type", "unet"),
                input_features=model.get("input_features", ["aod", "pm25", "pm10"]),
                hidden_layers=model.get("hidden_layers", [64, 128, 256, 128, 64]),
                activation=model.get("activation", "relu"),
                learning_rate=float(model.get("learning_rate", 0.001)),
                batch_size=int(model.get("batch_size", 32)),
                epochs=int(model.get("epochs", 100)),
                optimizer=model.get("optimizer", "adam"),
                loss_function=model.get("loss_function", "mse"),
                use_gpu=bool(model.get("use_gpu", True))
            )
            logger.debug(f"Loaded model config: {config}")
            return config
        except (KeyError, ValueError) as e:
            logger.error(f"Error loading model configuration: {e}")
            raise

    def get_data_config(self) -> DataConfig:
        """
        Get data configuration.

        Returns
        -------
        DataConfig
            Configuration dataclass for data sources and paths

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            data = self.config.get("data", {})
            config = DataConfig(
                ufs_data_path=data.get("ufs_data_path", "./data/ufs"),
                gefs_data_path=data.get("gefs_data_path", "./data/gefs"),
                output_path=data.get("output_path", "./output"),
                cache_dir=data.get("cache_dir", "~/.auroragcafs/cache"),
                historical_data=data.get("historical_data", None)
            )
            logger.debug(f"Loaded data config: {config}")
            return config
        except KeyError as e:
            logger.error(f"Error loading data configuration: {e}")
            raise

    def get_visualization_config(self) -> VisualizationConfig:
        """
        Get visualization configuration.

        Returns
        -------
        VisualizationConfig
            Configuration dataclass for visualization settings

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            viz = self.config.get("visualization", {})
            config = VisualizationConfig(
                map_projection=viz.get("map_projection", "PlateCarree"),
                colormap=viz.get("colormap", "viridis"),
                dpi=int(viz.get("dpi", 300)),
                figure_width=int(viz.get("figure_width", 10)),
                figure_height=int(viz.get("figure_height", 8))
            )
            logger.debug(f"Loaded visualization config: {config}")
            return config
        except (KeyError, ValueError) as e:
            logger.error(f"Error loading visualization configuration: {e}")
            raise

    def get_logging_config(self) -> LoggingConfig:
        """
        Get logging configuration.

        Returns
        -------
        LoggingConfig
            Configuration dataclass for logging settings

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            log = self.config.get("logging", {})
            config = LoggingConfig(
                level=log.get("level", "INFO"),
                format=log.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file=log.get("file", "logs/auroragcafs.log")
            )
            logger.debug(f"Loaded logging config: {config}")
            return config
        except KeyError as e:
            logger.error(f"Error loading logging configuration: {e}")
            raise

    def get_system_config(self) -> SystemConfig:
        """
        Get system configuration.

        Returns
        -------
        SystemConfig
            Configuration dataclass for system settings

        Raises
        ------
        KeyError
            If required configuration is missing
        """
        try:
            sys = self.config.get("system", {})
            config = SystemConfig(
                n_workers=int(sys.get("n_workers", 4)),
                memory_limit=sys.get("memory_limit", "8GB"),
                enable_dask=bool(sys.get("enable_dask", True))
            )
            logger.debug(f"Loaded system config: {config}")
            return config
        except (KeyError, ValueError) as e:
            logger.error(f"Error loading system configuration: {e}")
            raise

    def get_processing_config(self) -> ProcessingConfig:
        """
        Get the processing configuration.

        Returns
        -------
        ProcessingConfig
            Configuration for data processing

        Raises
        ------
        ValueError
            If required configuration values are missing or invalid
        """
        try:
            processing = self.config.get("processing", {})

            # Create normalization config
            norm_data = processing.get("normalization", {})

            # Create pre-log transform config
            log_transform_data = norm_data.get("pre_log_transform", {})
            log_transform_config = PreLogTransformConfig(
                enabled=log_transform_data.get("enabled", False),
                variables=log_transform_data.get("variables", []),
                base=log_transform_data.get("base", "e"),
                offset=float(log_transform_data.get("offset", 1.0))
            )

            # Create normalization config
            norm_config = NormalizationConfig(
                method=norm_data.get("method", "min_max"),
                per_variable=norm_data.get("per_variable", True),
                scale_factors=norm_data.get("scale_factors", {}),
                bounds=norm_data.get("bounds", {"min": 0.0, "max": 1.0}),
                epsilon=float(norm_data.get("epsilon", 1e-8)),
                pre_log_transform=log_transform_config
            )

            # Create processing config
            config = ProcessingConfig(
                variable_names=processing.get("variable_names", []),
                resolution=float(processing.get("resolution", 0.25)),
                temporal_frequency=processing.get("temporal_frequency", "6h"),
                interpolation_method=processing.get("interpolation_method", "bilinear"),
                normalization=norm_config
            )

            logger.debug(f"Loaded processing config: {config}")
            return config

        except (KeyError, ValueError) as e:
            logger.error(f"Error loading processing configuration: {e}")
            raise

    def save_config(self):
        """
        Save current configuration to file.

        Writes the current configuration to the YAML file.

        Raises
        ------
        IOError
            If writing fails
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                logger.info(f"Saved configuration to {self.config_file}")
        except IOError as e:
            logger.error(f"Error saving configuration: {e}")
            raise

# Create default config manager instance
config_manager = ConfigManager()

# Export commonly used configurations
get_aerosol_config = config_manager.get_aerosol_config
get_model_config = config_manager.get_model_config
get_data_config = config_manager.get_data_config
get_processing_config = config_manager.get_processing_config
get_visualization_config = config_manager.get_visualization_config
get_logging_config = config_manager.get_logging_config
get_system_config = config_manager.get_system_config
get_processing_config = config_manager.get_processing_config