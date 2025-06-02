#!/usr/bin/env python3
# filepath: /Users/l22-n04127-res/Documents/GitHub/AuroraGCAFS/scripts/evaluate_model.py
"""
Model evaluation script for AuroraGCAFS

This script evaluates a trained model on test data and generates a comprehensive
set of evaluation metrics and performance visualizations.

Usage:
    python evaluate_model.py --model-path <model_path> --data-start <YYYYMMDD> --data-end <YYYYMMDD> --output-dir <output_dir>

Author: NOAA/NESDIS Team
Date: June 2025
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import json
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auroragcafs.data import DataProcessor
from auroragcafs.model import AerosolModel
from auroragcafs.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AuroraGCAFS model")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data-start', type=str, required=True,
                        help='Start date for evaluation data (YYYYMMDD format)')
    parser.add_argument('--data-end', type=str, required=True,
                        help='End date for evaluation data (YYYYMMDD format)')
    parser.add_argument('--config-path', type=str,
                        default='/Users/l22-n04127-res/Documents/GitHub/AuroraGCAFS/parm/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loader')
    return parser.parse_args()

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data and return predictions and metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(all_targets, all_preds))),
        'mae': float(mean_absolute_error(all_targets, all_preds)),
        'r2': float(r2_score(all_targets, all_preds)),
        'mean_bias': float(np.mean(all_preds - all_targets)),
        'mean_abs_error': float(np.mean(np.abs(all_preds - all_targets)))
    }

    # Calculate per-variable metrics if available
    if hasattr(test_loader.dataset, 'feature_names') and hasattr(test_loader.dataset, 'target_names'):
        var_metrics = {}
        for i, var_name in enumerate(test_loader.dataset.target_names):
            var_preds = all_preds[:, i]
            var_targets = all_targets[:, i]
            var_metrics[var_name] = {
                'rmse': float(np.sqrt(mean_squared_error(var_targets, var_preds))),
                'mae': float(mean_absolute_error(var_targets, var_preds)),
                'r2': float(r2_score(var_targets, var_preds)),
                'mean_bias': float(np.mean(var_preds - var_targets))
            }
        metrics['per_variable'] = var_metrics

    return all_preds, all_targets, metrics

def generate_visualizations(predictions, targets, dataset, output_dir):
    """Generate and save visualizations of model performance."""
    os.makedirs(output_dir, exist_ok=True)

    # Create scatter plots for each target variable
    if hasattr(dataset, 'target_names'):
        for i, var_name in enumerate(dataset.target_names):
            plt.figure(figsize=(10, 10))
            plt.scatter(targets[:, i], predictions[:, i], alpha=0.3)

            # Add perfect prediction line
            min_val = min(np.min(targets[:, i]), np.min(predictions[:, i]))
            max_val = max(np.max(targets[:, i]), np.max(predictions[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')

            plt.xlabel(f"True {var_name}")
            plt.ylabel(f"Predicted {var_name}")
            plt.title(f"True vs. Predicted {var_name}")

            # Add performance metrics to plot
            rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
            r2 = r2_score(targets[:, i], predictions[:, i])
            plt.annotate(f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        verticalalignment='top')

            plt.savefig(os.path.join(output_dir, f"scatter_{var_name}.png"))
            plt.close()

    # Create error distribution plots
    errors = predictions - targets
    plt.figure(figsize=(12, 8))
    sns.histplot(errors.flatten(), kde=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging file handler
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log'))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Starting evaluation of model: {args.model_path}")
    logger.info(f"Evaluation period: {args.data_start} to {args.data_end}")

    # Load configuration
    config = load_config(args.config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize data processor
    data_processor = DataProcessor(config)

    # Load test dataset
    try:
        start_date = datetime.strptime(args.data_start, "%Y%m%d")
        end_date = datetime.strptime(args.data_end, "%Y%m%d")

        logger.info("Loading test dataset...")
        test_dataset = data_processor.create_dataset(
            start_date=start_date,
            end_date=end_date,
            mode='test'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        logger.info(f"Test dataset loaded with {len(test_dataset)} samples")

    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        sys.exit(1)

    # Load model
    try:
        logger.info(f"Loading model from {args.model_path}...")
        model = AerosolModel(config)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Evaluate model
    logger.info("Evaluating model...")
    try:
        predictions, targets, metrics = evaluate_model(model, test_loader, device)
        logger.info(f"Overall RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Overall MAE: {metrics['mae']:.4f}")
        logger.info(f"Overall R²: {metrics['r2']:.4f}")

        # Save metrics to JSON file
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # Generate visualizations
        logger.info("Generating visualizations...")
        viz_dir = os.path.join(args.output_dir, "visualizations")
        generate_visualizations(predictions, targets, test_dataset, viz_dir)
        logger.info(f"Visualizations saved to {viz_dir}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()
