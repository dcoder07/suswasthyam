"""
Demonstration script for the health prediction model.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from data_processor import DataProcessor
from model import HealthPredictionModel
from utils.helpers import generate_synthetic_data, plot_vital_signs
import config

def run_complete_demo(output_dir=None):
    """
    Run a complete demonstration of the health prediction model.
    
    Args:
        output_dir (str): Directory to save output files
    """
    if output_dir is None:
        output_dir = f"demo_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Demo outputs will be saved to {output_dir}")
    
    # 1. Generate synthetic training data
    print("\n1. Generating synthetic training data...")
    train_data = generate_synthetic_data(
        n_days=30, 
        readings_per_day=4,
        abnormal_days=[5, 6, 7, 15, 16, 22, 23, 24],
        output_path=os.path.join(output_dir, "training_data.csv")
    )
    
    plot_vital_signs(train_data, 
                    output_path=os.path.join(output_dir, "training_data_plot.png"))
    
    # 2. Train the model
    print("\n2. Training the model...")
    from train import train_model
    
    model, metrics = train_model(
        os.path.join(output_dir, "training_data.csv"),
        os.path.join(output_dir, "health_model.h5"),
        visualize=True
    )
    
    print("\nTraining complete! Model evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 3. Generate new test data
    print("\n3. Generating test data...")
    test_data = generate_synthetic_data(
        n_days=10, 
        readings_per_day=4,
        abnormal_days=[3, 7],
        output_path=os.path.join(output_dir, "test_data.csv")
    )
    
    plot_vital_signs(test_data, 
                    output_path=os.path.join(output_dir, "test_data_plot.png"))
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    from predict import predict_health_status
    
    predictions = predict_health_status(
        os.path.join(output_dir, "test_data.csv"),
        os.path.join(output_dir, "health_model.h5"),
        days_ahead=7,
        output_path=os.path.join(output_dir, "predictions.csv"),
        visualize=True
    )
    
    # 5. Fine-tune the model with new data
    print("\n5. Fine-tuning the model with new data...")
    from train import fine_tune_model
    
    fine_tuned_model, ft_metrics = fine_tune_model(
        os.path.join(output_dir, "test_data.csv"),
        os.path.join(output_dir, "health_model.h5"),
        visualize=True
    )
    
    if ft_metrics:
        print("\nFine-tuning complete! Updated model evaluation metrics:")
        for metric, value in ft_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # 6. Generate future data and evaluate
    print("\n6. Generating future data to test improvement...")
    future_data = generate_synthetic_data(
        n_days=10, 
        readings_per_day=4,
        abnormal_days=[2, 4, 8],
        output_path=os.path.join(output_dir, "future_data.csv")
    )
    
    # 7. Evaluate the fine-tuned model
    print("\n7. Evaluating the fine-tuned model...")
    from evaluate import evaluate_model
    
    eval_metrics = evaluate_model(
        os.path.join(output_dir, "future_data.csv"),
        os.path.join(output_dir, "health_model.h5"),
        output_dir=os.path.join(output_dir, "evaluation"),
        visualize=True
    )
    
    print("\nDemo complete! All outputs have been saved to:", output_dir)
    print("You can now examine the model's performance and predictions.")

def main():
    """Main function for running the demo."""
    parser = argparse.ArgumentParser(description="Run a demonstration of the health prediction model")
    parser.add_argument("--output_dir", help="Directory to save output files")
    
    args = parser.parse_args()
    
    run_complete_demo(args.output_dir)

if __name__ == "__main__":
    main() 