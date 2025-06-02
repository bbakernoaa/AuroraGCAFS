"""
Model module for AuroraGCAFS.

This module provides model architectures and training utilities for
processing aerosol data using deep learning approaches.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from ..config import config_manager, ModelConfig

# Set up module logger
logger = logging.getLogger(__name__)

class AerosolDataset(Dataset):
    """
    Dataset class for aerosol data.

    Parameters
    ----------
    input_data : xr.Dataset
        Input dataset containing aerosol variables
    target_data : xr.Dataset, optional
        Target dataset for supervised learning
    transform : callable, optional
        Transform to apply to the data

    Attributes
    ----------
    input_data : xr.Dataset
        Input dataset
    target_data : xr.Dataset
        Target dataset
    transform : callable
        Transform function
    """

    def __init__(self,
                 input_data: xr.Dataset,
                 target_data: Optional[xr.Dataset] = None,
                 transform: Optional[callable] = None):
        self.input_data = input_data
        self.target_data = target_data
        self.transform = transform
        logger.debug(f"AerosolDataset initialized with {len(input_data)} samples")

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        dict
            Dictionary containing input and target tensors
        """
        sample = {'input': torch.from_numpy(self.input_data[idx].values)}

        if self.target_data is not None:
            sample['target'] = torch.from_numpy(self.target_data[idx].values)

        if self.transform:
            sample = self.transform(sample)

        return sample

class UNet(nn.Module):
    """
    U-Net architecture for aerosol data processing.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    hidden_channels : List[int]
        List of channel numbers in hidden layers
    activation : str
        Activation function to use

    Attributes
    ----------
    encoder : nn.ModuleList
        Encoder layers
    decoder : nn.ModuleList
        Decoder layers
    final_conv : nn.Conv2d
        Final convolution layer
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: List[int] = [64, 128, 256, 512],
                 activation: str = "relu"):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder path
        channels = in_channels
        for hidden in hidden_channels:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 3, padding=1),
                    nn.BatchNorm2d(hidden),
                    self._get_activation(activation),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.BatchNorm2d(hidden),
                    self._get_activation(activation)
                )
            )
            channels = hidden

        # Decoder path
        for hidden in reversed(hidden_channels[:-1]):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, hidden, 3, padding=1),
                    nn.BatchNorm2d(hidden),
                    self._get_activation(activation),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.BatchNorm2d(hidden),
                    self._get_activation(activation)
                )
            )
            channels = hidden

        self.final_conv = nn.Conv2d(channels, out_channels, 1)
        logger.info(f"UNet initialized with {len(hidden_channels)} layers")

    def _get_activation(self, name: str) -> nn.Module:
        """
        Get activation function by name.

        Parameters
        ----------
        name : str
            Name of activation function

        Returns
        -------
        nn.Module
            Activation function

        Raises
        ------
        ValueError
            If activation function is not supported
        """
        if name.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif name.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        elif name.lower() == "elu":
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            encoder_outputs.append(x)
            x = F.max_pool2d(x, 2)

        # Decoder path
        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_outputs[-(i+2)]], dim=1)
            x = dec(x)

        return self.final_conv(x)

class AerosolModel:
    """
    Main model class for AuroraGCAFS with continual learning support.

    This class handles model creation, training, and inference.

    Parameters
    ----------
    config : ModelConfig, optional
        Model configuration
    device : str, optional
        Device to use for computations

    Attributes
    ----------
    config : ModelConfig
        Model configuration
    device : torch.device
        Computation device
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Model optimizer
    criterion : nn.Module
        Loss function
    fisher_info : dict
        Fisher Information Matrix for EWC
    old_params : dict
        Parameters from previous training for EWC
    """

    def __init__(self,
                 config: Optional[ModelConfig] = None,
                 device: Optional[str] = None):
        self.config = config if config else config_manager.get_model_config()

        if device is None:
            device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        self.device = torch.device(device)

        self.model = self._create_model()
        self.model.to(self.device)

        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_function()

        self.fisher_info = None
        self.old_params = None

        logger.info(f"AerosolModel initialized on device: {self.device}")

    def _create_model(self) -> nn.Module:
        """
        Create the neural network model.

        Returns
        -------
        nn.Module
            Neural network model

        Raises
        ------
        ValueError
            If model type is not supported
        """
        if self.config.model_type.lower() == "unet":
            return UNet(
                in_channels=len(self.config.input_features),
                out_channels=len(self.config.input_features),
                hidden_channels=self.config.hidden_layers,
                activation=self.config.activation
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def compute_fisher_information(self, train_loader: DataLoader):
        """
        Compute Fisher Information Matrix for EWC.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data

        Returns
        -------
        dict
            Fisher Information Matrix for each parameter
        """
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for batch in train_loader:
            self.model.zero_grad()
            output = self.model(batch['input'].to(self.device))
            loss = self.criterion(output, batch['target'].to(self.device))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_info[n] += p.grad.pow(2).detach()

        # Normalize by number of samples
        for n in fisher_info:
            fisher_info[n] /= len(train_loader)

        return fisher_info

    def prepare_continual_learning(self, train_loader: DataLoader):
        """
        Prepare model for continual learning by computing Fisher information.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        """
        self.fisher_info = self.compute_fisher_information(train_loader)
        self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def compute_ewc_loss(self, lambda_ewc: float = 100.0) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Parameters
        ----------
        lambda_ewc : float
            Regularization strength

        Returns
        -------
        torch.Tensor
            EWC loss
        """
        if self.fisher_info is None or self.old_params is None:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            ewc_loss += (self.fisher_info[n] * (p - self.old_params[n]).pow(2)).sum()

        return (lambda_ewc / 2) * ewc_loss

    def train(self,
              train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None,
              lambda_ewc: float = 100.0,
              **kwargs: Any) -> Dict[str, List[float]]:
        """
        Train the model with support for continual learning.

        Parameters
        ----------
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset, optional
            Validation dataset
        lambda_ewc : float
            EWC regularization strength
        **kwargs : dict
            Additional arguments to pass to training loop

        Returns
        -------
        dict
            Dictionary containing training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

        history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_ewc_loss': [],
            'val_loss': [] if val_loader else None
        }

        logger.info("Starting model training")
        self.model.train()

        for epoch in range(self.config.epochs):
            train_loss = 0.0
            train_task_loss = 0.0
            train_ewc_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()

                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                outputs = self.model(inputs)

                # Compute main task loss
                task_loss = self.criterion(outputs, targets)

                # Add EWC regularization if available
                ewc_loss = self.compute_ewc_loss(lambda_ewc)
                loss = task_loss + ewc_loss

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_task_loss += task_loss.item()
                train_ewc_loss += ewc_loss.item()

            train_loss /= len(train_loader)
            train_task_loss /= len(train_loader)
            train_ewc_loss /= len(train_loader)

            history['train_loss'].append(train_loss)
            history['train_task_loss'].append(train_task_loss)
            history['train_ewc_loss'].append(train_ewc_loss)

            if val_loader:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Train Loss: {train_loss:.4f} "
                            f"(Task: {train_task_loss:.4f}, EWC: {train_ewc_loss:.4f}) - "
                            f"Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Train Loss: {train_loss:.4f} "
                            f"(Task: {train_task_loss:.4f}, EWC: {train_ewc_loss:.4f})")

        return history

    def _validate(self,
                 val_loader: DataLoader,
                 criterion: nn.Module) -> float:
        """
        Validate the model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function

        Returns
        -------
        float
            Validation loss
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        self.model.train()
        return val_loss / len(val_loader)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance

        Raises
        ------
        ValueError
            If optimizer is not supported
        """
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _get_loss_function(self) -> nn.Module:
        """
        Get the loss function.

        Returns
        -------
        nn.Module
            Loss function

        Raises
        ------
        ValueError
            If loss function is not supported
        """
        if self.config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_function.lower() == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def save(self, path: str):
        """
        Save the model.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load the model.

        Parameters
        ----------
        path : str
            Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

# Create default model instance
model = AerosolModel()

__all__ = ['AerosolDataset', 'UNet', 'AerosolModel', 'model']