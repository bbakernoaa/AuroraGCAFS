#!/usr/bin/env python3
# filepath: /Users/l22-n04127-res/Documents/GitHub/AuroraGCAFS/scripts/update_metrics.py
"""
Metrics Update and Tracking script for AuroraGCAFS

This script collects and aggregates model evaluation metrics across multiple
training periods and generates reports on model improvement trends.

Usage:
    python update_metrics.py --metrics-dir <dir> --output-dir <output_dir>

Author: NOAA/NESDIS Team
Date: June 2025
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
import glob
from matplotlib.ticker import MaxNLocator
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Update and track AuroraGCAFS model metrics")
    parser.add_argument('--metrics-dir', type=str, required=True,
                        help='Directory containing model evaluation metrics')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save aggregated metrics and reports')
    return parser.parse_args()

def collect_metrics(metrics_dir):
    """Collect and aggregate metrics from the metrics directory."""
    metrics_files = glob.glob(os.path.join(metrics_dir, "**/metrics.json"), recursive=True)

    if not metrics_files:
        logger.warning(f"No metrics files found in {metrics_dir}")
        return None

    metrics_data = []

    for metrics_file in metrics_files:
        # Extract date from the metrics file path using regex
        date_match = re.search(r'(\d{8})_(\d{8})', os.path.dirname(metrics_file))
        if not date_match:
            logger.warning(f"Could not extract date from metrics file path: {metrics_file}")
            continue

        start_date = date_match.group(1)
        end_date = date_match.group(2)

        # Load metrics from JSON file
        with open(metrics_file, 'r') as f:
            try:
                metrics = json.load(f)

                # Add date information to metrics
                metrics['start_date'] = start_date
                metrics['end_date'] = end_date
                metrics_data.append(metrics)

            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from file: {metrics_file}")

    # Sort metrics by date
    metrics_data.sort(key=lambda x: x['start_date'])
    return metrics_data

def generate_metrics_df(metrics_data):
    """Generate a DataFrame from metrics data."""
    metrics_df = pd.DataFrame({
        'start_date': [m['start_date'] for m in metrics_data],
        'end_date': [m['end_date'] for m in metrics_data],
        'rmse': [m['rmse'] for m in metrics_data],
        'mae': [m['mae'] for m in metrics_data],
        'r2': [m['r2'] for m in metrics_data],
        'mean_bias': [m['mean_bias'] for m in metrics_data]
    })

    # Convert date strings to datetime objects
    metrics_df['start_date'] = pd.to_datetime(metrics_df['start_date'], format='%Y%m%d')
    metrics_df['end_date'] = pd.to_datetime(metrics_df['end_date'], format='%Y%m%d')

    return metrics_df

def generate_trend_plots(metrics_df, output_dir):
    """Generate trend plots for metrics over time."""
    os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Generate RMSE trend plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['start_date'], metrics_df['rmse'], marker='o', linewidth=2)
    plt.xlabel('Training Period')
    plt.ylabel('RMSE')
    plt.title('Model RMSE Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_trend.png'))
    plt.close()

    # Generate R² trend plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['start_date'], metrics_df['r2'], marker='o', linewidth=2, color='green')
    plt.xlabel('Training Period')
    plt.ylabel('R²')
    plt.title('Model R² Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_trend.png'))
    plt.close()

    # Generate MAE trend plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['start_date'], metrics_df['mae'], marker='o', linewidth=2, color='orange')
    plt.xlabel('Training Period')
    plt.ylabel('MAE')
    plt.title('Model MAE Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_trend.png'))
    plt.close()

    # Generate mean bias trend plot
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['start_date'], metrics_df['mean_bias'], marker='o', linewidth=2, color='red')
    plt.xlabel('Training Period')
    plt.ylabel('Mean Bias')
    plt.title('Model Mean Bias Trend Over Time')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_bias_trend.png'))
    plt.close()

    # Generate combined metrics plot
    fig, ax1 = plt.subplots(figsize=(14, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Training Period')
    ax1.set_ylabel('RMSE / MAE', color=color)
    ax1.plot(metrics_df['start_date'], metrics_df['rmse'], marker='o', linewidth=2,
             label='RMSE', color='tab:blue')
    ax1.plot(metrics_df['start_date'], metrics_df['mae'], marker='s', linewidth=2,
             label='MAE', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('R²', color=color)
    ax2.plot(metrics_df['start_date'], metrics_df['r2'], marker='^', linewidth=2,
             label='R²', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Model Performance Metrics Over Time')
    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()

def generate_summary_report(metrics_df, output_dir):
    """Generate a summary report of model improvements."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate metrics improvements
    first_metrics = metrics_df.iloc[0]
    last_metrics = metrics_df.iloc[-1]

    rmse_change = last_metrics['rmse'] - first_metrics['rmse']
    rmse_pct_change = (rmse_change / first_metrics['rmse']) * 100

    mae_change = last_metrics['mae'] - first_metrics['mae']
    mae_pct_change = (mae_change / first_metrics['mae']) * 100

    r2_change = last_metrics['r2'] - first_metrics['r2']
    r2_pct_change = (r2_change / first_metrics['r2']) * 100 if first_metrics['r2'] != 0 else 0

    # Create summary report
    report = {
        'start_period': first_metrics['start_date'].strftime('%Y-%m-%d'),
        'end_period': last_metrics['end_date'].strftime('%Y-%m-%d'),
        'num_periods': len(metrics_df),
        'initial_metrics': {
            'rmse': first_metrics['rmse'],
            'mae': first_metrics['mae'],
            'r2': first_metrics['r2'],
            'mean_bias': first_metrics['mean_bias']
        },
        'final_metrics': {
            'rmse': last_metrics['rmse'],
            'mae': last_metrics['mae'],
            'r2': last_metrics['r2'],
            'mean_bias': last_metrics['mean_bias']
        },
        'improvements': {
            'rmse': {
                'absolute': -rmse_change,  # Negative change is improvement for RMSE
                'percentage': -rmse_pct_change
            },
            'mae': {
                'absolute': -mae_change,  # Negative change is improvement for MAE
                'percentage': -mae_pct_change
            },
            'r2': {
                'absolute': r2_change,  # Positive change is improvement for R²
                'percentage': r2_pct_change
            }
        }
    }

    # Save report as JSON
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Generate text report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("AuroraGCAFS Model Performance Summary\n")
        f.write("===================================\n\n")
        f.write(f"Training Period: {report['start_period']} to {report['end_period']}\n")
        f.write(f"Number of Training Periods: {report['num_periods']}\n\n")

        f.write("Initial Metrics:\n")
        f.write(f"  RMSE: {report['initial_metrics']['rmse']:.4f}\n")
        f.write(f"  MAE: {report['initial_metrics']['mae']:.4f}\n")
        f.write(f"  R²: {report['initial_metrics']['r2']:.4f}\n")
        f.write(f"  Mean Bias: {report['initial_metrics']['mean_bias']:.4f}\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  RMSE: {report['final_metrics']['rmse']:.4f}\n")
        f.write(f"  MAE: {report['final_metrics']['mae']:.4f}\n")
        f.write(f"  R²: {report['final_metrics']['r2']:.4f}\n")
        f.write(f"  Mean Bias: {report['final_metrics']['mean_bias']:.4f}\n\n")

        f.write("Improvements:\n")
        f.write(f"  RMSE: {report['improvements']['rmse']['absolute']:.4f} ")
        f.write(f"({report['improvements']['rmse']['percentage']:.2f}%)\n")

        f.write(f"  MAE: {report['improvements']['mae']['absolute']:.4f} ")
        f.write(f"({report['improvements']['mae']['percentage']:.2f}%)\n")

        f.write(f"  R²: {report['improvements']['r2']['absolute']:.4f} ")
        f.write(f"({report['improvements']['r2']['percentage']:.2f}%)\n\n")

        f.write("Report generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    return report

def main():
    # Parse arguments
    args = parse_args()

    # Set up logging file handler
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'metrics_update.log'))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Starting metrics update process")

    # Collect metrics data
    logger.info(f"Collecting metrics from {args.metrics_dir}")
    metrics_data = collect_metrics(args.metrics_dir)

    if not metrics_data:
        logger.error("No metrics data found. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(metrics_data)} metrics entries")

    # Generate metrics DataFrame
    metrics_df = generate_metrics_df(metrics_data)

    # Save metrics DataFrame to CSV
    metrics_csv = os.path.join(args.output_dir, 'metrics_history.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Metrics history saved to {metrics_csv}")

    # Generate trend plots
    logger.info("Generating trend plots")
    plots_dir = os.path.join(args.output_dir, 'plots')
    generate_trend_plots(metrics_df, plots_dir)
    logger.info(f"Trend plots saved to {plots_dir}")

    # Generate summary report
    logger.info("Generating summary report")
    report = generate_summary_report(metrics_df, args.output_dir)
    logger.info("Summary report generated")

    # Log summary of improvements
    for metric in ['rmse', 'mae', 'r2']:
        improvement = report['improvements'][metric]['percentage']
        direction = "improved" if (metric == 'r2' and improvement > 0) or \
                                (metric != 'r2' and improvement < 0) else "degraded"
        logger.info(f"{metric.upper()}: {abs(improvement):.2f}% {direction}")

    logger.info("Metrics update process completed successfully")

if __name__ == "__main__":
    main()
