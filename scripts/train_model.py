#!/usr/bin/env python3
"""
Training script for AuroraGCAFS with support for continual learning.

This script provides functionality to train the AuroraGCAFS model either from
scratch or incrementally on new data while preserving knowledge from previous
training sessions. It implements Elastic Weight Consolidation (EWC) to prevent
catastrophic forgetting during continual learning.

Usage:
    python train_model.py --start-date YYYYMMDD --end-date YYYYMMDD [options]

Examples:
    # Train from scratch
    python train_model.py --start-date 20250101 --end-date 20250131 --output-dir ./outputs

    # Continue training with a pre-trained model
    python train_model.py --start-date 20250201 --end-date 20250228 --model-path model.pt --output-dir ./outputs

Author: NOAA/NESDIS Team
Date: May 2025
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import multiprocessing as mp
from typing import Optional

import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from auroragcafs import config_manager, ufs_loader, gefs_loader, processor
from auroragcafs.model import AerosolModel, AerosolDataset

logger = logging.getLogger(__name__)

def setup_argparse():
    """
    Set up command line argument parsing.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Train AuroraGCAFS model')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date in YYYYMMDD format')
    parser.add_argument('--model-path', type=str,
                       help='Path to load pre-trained model for continual learning')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per training session')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--ewc-lambda', type=float, default=100.0,
                       help='EWC regularization strength for preventing catastrophic forgetting')
    return parser

def _load_and_process_day(date: pd.Timestamp) -> Optional[xr.Dataset]:
    """
    Loads and processes UFS forecast data for a single day.

    Parameters
    ----------
    date : pd.Timestamp
        The date for which to load data.

    Returns
    -------
    Optional[xr.Dataset]
        The processed xarray Dataset, or None if loading fails.
    """
    date_str = date.strftime('%Y%m%d')
    try:
        ds = ufs_loader.load_forecast(date_str, 0)
        ds = processor.normalize(ds)
        return ds
    except Exception as e:
        logger.warning(f"Failed to load UFS data for {date_str}: {e}")
        return None

def _get_data_catalog(start_date: str, end_date: str) -> pd.Series:
    """
    Get a catalog of data files to be loaded.

    Parameters
    ----------
    start_date : str
        Start date in YYYYMMDD format.
    end_date : str
        End date in YYYYMMDD format.

    Returns
    -------
    pd.Series
        A series of dates to be loaded.
    """
    return pd.date_range(start_date, end_date, freq='D')

def _load_data_parallel(dates: pd.Series) -> list:
    """
    Load data in parallel using multiprocessing.

    Parameters
    ----------
    dates : pd.Series
        A series of dates to be loaded.

    Returns
    -------
    list
        A list of loaded xarray Datasets.
    """
    ufs_data = []
    with mp.Pool(mp.cpu_count()) as pool:
        with tqdm(total=len(dates), desc="Loading UFS Data") as pbar:
            for result in pool.imap_unordered(_load_and_process_day, dates):
                if result is not None:
                    ufs_data.append(result)
                pbar.update()
    return ufs_data

def _create_datasets(ufs_data: list) -> tuple:
    """
    Create training and validation datasets from a list of xarray Datasets.

    Parameters
    ----------
    ufs_data : list
        A list of loaded xarray Datasets.

    Returns
    -------
    tuple
        A tuple containing (train_dataset, val_dataset) for model training.
    """
    if not ufs_data:
        raise ValueError("No valid training data found")

    # Sort data by time to ensure correct chronological order before concatenation
    ufs_data.sort(key=lambda ds: ds.time.values[0])

    # Combine all data
    combined_data = xr.concat(ufs_data, dim='time')

    # Split into train/val
    n_samples = len(combined_data.time)
    train_size = int(0.8 * n_samples)

    train_data = combined_data.isel(time=slice(0, train_size))
    val_data = combined_data.isel(time=slice(train_size, None))

    # Create datasets
    train_dataset = AerosolDataset(train_data)
    val_dataset = AerosolDataset(val_data)

    return train_dataset, val_dataset

def prepare_data(start_date: str, end_date: str, config) -> tuple:
    """
    Prepare training and validation datasets from UFS data.

    This function loads UFS aerosol data for the specified date range in parallel,
    normalizes it using the configured normalization method, and splits it
    into training and validation datasets.

    Parameters
    ----------
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
    config : Config
        Configuration manager instance

    Returns
    -------
    tuple
        A tuple containing (train_dataset, val_dataset) for model training

    Raises
    ------
    ValueError
        If no valid training data is found in the date range
    """
    logger.info("Starting parallel data preparation...")
    start_time = time.perf_counter()

    dates = _get_data_catalog(start_date, end_date)
    ufs_data = _load_data_parallel(dates)
    train_dataset, val_dataset = _create_datasets(ufs_data)

    end_time = time.perf_counter()
    logger.info(f"Data preparation finished in {end_time - start_time:.2f} seconds.")

    return train_dataset, val_dataset

def compute_fisher_information(model: AerosolModel, train_loader: DataLoader):
    """
    Compute Fisher Information Matrix for Elastic Weight Consolidation (EWC).

    This function calculates the Fisher Information Matrix, which represents
    the importance of each model parameter for the current task. This information
    is used by EWC to prevent catastrophic forgetting during continual learning.

    Parameters
    ----------
    model : AerosolModel
        The trained model
    train_loader : DataLoader
        DataLoader containing training data

    Returns
    -------
    dict
        Dictionary mapping parameter names to their Fisher information
    """
    model.model.eval()
    fisher_info = {n: torch.zeros_like(p) for n, p in model.model.named_parameters()}

    for batch in train_loader:
        model.model.zero_grad()
        output = model.model(batch['input'].to(model.device))
        loss = model.criterion(output, batch['target'].to(model.device))
        loss.backward()

        for n, p in model.model.named_parameters():
            if p.grad is not None:
                fisher_info[n] += p.grad.pow(2).detach()

    # Normalize by number of samples
    for n in fisher_info:
        fisher_info[n] /= len(train_loader)

    return fisher_info

def main():
    """
    Main entry point for the training script.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Prepares training and validation data
    4. Initializes or loads the model
    5. Trains the model with EWC if continuing from a pre-trained model
    6. Saves checkpoints and the final model

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    parser = setup_argparse()
    args = parser.parse_args()

    # Set up logging
    log_dir = Path(args.output_dir) / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{args.start_date}_{args.end_date}.log'),
            logging.StreamHandler()
        ]
    )

    # Prepare data
    train_dataset, val_dataset = prepare_data(args.start_date, args.end_date, config_manager)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize or load model
    model = AerosolModel()
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading pre-trained model from {args.model_path}")
        model.load(args.model_path)

        # Compute Fisher information for EWC
        fisher_info = compute_fisher_information(model, train_loader)
        old_params = {n: p.clone().detach() for n, p in model.model.named_parameters()}
    else:
        fisher_info = None
        old_params = None

    # Training loop with EWC
    checkpoints_dir = Path(args.output_dir) / 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch['input'].to(model.device)
            targets = batch['target'].to(model.device)

            model.optimizer.zero_grad()
            outputs = model.model(inputs)

            # Compute main loss
            loss = model.criterion(outputs, targets)

            # Add EWC regularization if doing continual learning
            if fisher_info is not None and old_params is not None:
                ewc_loss = 0
                for n, p in model.model.named_parameters():
                    _lambda = args.ewc_lambda
                    ewc_loss += (fisher_info[n] * (p - old_params[n]).pow(2)).sum()
                loss += (_lambda / 2) * ewc_loss

            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()

        # Validation
        model.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(model.device)
                targets = batch['target'].to(model.device)
                outputs = model.model(inputs)
                val_loss += model.criterion(outputs, targets).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f} - "
                   f"Val Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoints_dir / f"model_{args.start_date}_{args.end_date}_best.pt"
            model.save(checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")

    # Save final model
    final_path = checkpoints_dir / f"model_{args.start_date}_{args.end_date}_final.pt"
    model.save(final_path)
    logger.info(f"Saved final model to {final_path}")

if __name__ == '__main__':
    main()
