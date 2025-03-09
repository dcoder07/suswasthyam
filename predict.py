"""
Prediction script for the health prediction model.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data_processor import DataProcessor
from model import HealthPredictionModel
import config

def predict_health_status(data_path, model_path=None, days_ahead=7, output_path=None, visualize=True):
    """
    Make health status predictions based on vital signs data.
    
    Args:
        data_path (str): Path to the CSV file with input data
        model_path (str): Path to the saved model
        days_ahead (int): Number of days to predict ahead
        output_path (str): Path to save the prediction results
        visualize (bool): Whether to visualize prediction results
        
    Returns:
        pd.DataFrame: DataFrame with prediction results
    """
    # Initialize model and load weights
    model = HealthPredictionModel()
    if not model.load(model_path):
        raise FileNotFoundError(f"No model found at {model_path or config.MODEL_SAVE_PATH}")
    
    # Initialize data processor and prepare data
    data_processor = DataProcessor()
    data_processor.load_scaler()
    
    # Prepare prediction data
    X, timestamps = data_processor.prepare_prediction_data(data_path)
    
    if len(X) == 0:
        raise ValueError("Not enough data points for prediction")
    
    # Make predictions for the existing data points
    predictions = model.predict(X).flatten()
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_health_status': predictions
    })
    
    # Round predictions for binary classification
    results['predicted_class'] = (results['predicted_health_status'] > 0.5).astype(int)
    
    # Load original data to get the original features
    original_data = data_processor.load_data(data_path)
    
    # Match predictions with original data based on timestamps
    results = pd.merge(
        results, 
        original_data[config.FEATURES + [config.TIMESTAMP_COL]], 
        left_on='timestamp',
        right_on=config.TIMESTAMP_COL,
        how='left'
    )
    
    # Make future predictions if requested
    if days_ahead > 0:
        future_predictions = model.predict_for_days(X, days=days_ahead)
        
        # Create future timestamps (daily predictions)
        # Convert last_timestamp to datetime if it's not already
        last_timestamp = timestamps[-1]
        if isinstance(last_timestamp, np.datetime64):
            last_timestamp = pd.Timestamp(last_timestamp).to_pydatetime()
            
        future_timestamps = [
            last_timestamp + timedelta(days=i+1) 
            for i in range(days_ahead)
        ]
        
        # Create future results DataFrame
        future_results = pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_health_status': future_predictions,
            'predicted_class': [int(p > 0.5) for p in future_predictions],
            'is_future_prediction': [True] * days_ahead
        })
        
        # Mark existing predictions as not future
        results['is_future_prediction'] = False
        
        # Combine existing and future predictions
        results = pd.concat([results, future_results], ignore_index=True)
    
    # Save results to CSV if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    # Visualize results if requested
    if visualize:
        visualize_predictions(results, days_ahead)
    
    return results

def visualize_predictions(results, days_ahead=0):
    """
    Visualize prediction results.
    
    Args:
        results (pd.DataFrame): DataFrame with prediction results
        days_ahead (int): Number of days predicted ahead
    """
    plt.figure(figsize=(12, 8))
    
    # Separate current and future data if available
    if 'is_future_prediction' in results.columns and days_ahead > 0:
        current_data = results[~results['is_future_prediction']]
        future_data = results[results['is_future_prediction']]
        
        # Plot boundary line between actual and predicted days
        plt.axvline(x=len(current_data)-0.5, color='gray', linestyle='--', 
                   label='Current/Future Boundary')
    else:
        current_data = results
        future_data = pd.DataFrame()
    
    # Plot health status prediction probabilities
    plt.subplot(2, 1, 1)
    plt.plot(range(len(current_data)), current_data['predicted_health_status'], 
             marker='o', label='Current Predictions')
    
    if not future_data.empty:
        plt.plot(range(len(current_data), len(results)), 
                 future_data['predicted_health_status'], 
                 marker='*', linestyle='--', color='red', 
                 label='Future Predictions')
        
        # Plot boundary line between actual and predicted days
        plt.axvline(x=len(current_data)-0.5, color='gray', linestyle='--')
    
    plt.axhline(y=0.5, color='green', linestyle=':',
                label='Classification Threshold')
    plt.title('Health Status Predictions')
    plt.ylabel('Probability of Health Issue')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot vital signs if available
    if all(feat in results.columns for feat in config.FEATURES):
        plt.subplot(2, 1, 2)
        
        # Plot temperature
        plt.plot(range(len(current_data)), current_data['temperature'], 
                marker='o', label='Temperature')
        
        # Add a second y-axis for SpO2 and heart rate
        ax2 = plt.twinx()
        ax2.plot(range(len(current_data)), current_data['spo2'], 
                marker='s', color='green', label='SpO2')
        ax2.plot(range(len(current_data)), current_data['heart_rate'] / 2,  # Scale heart rate for visibility
                marker='^', color='red', label='Heart Rate (scaled)')
        
        # Add future predictions if available
        if not future_data.empty and 'temperature' in future_data.columns:
            plt.plot(range(len(current_data), len(results)), 
                     future_data['temperature'], 
                     linestyle='--', color='blue')
            
            if 'spo2' in future_data.columns and 'heart_rate' in future_data.columns:
                ax2.plot(range(len(current_data), len(results)), 
                         future_data['spo2'], 
                         linestyle='--', color='green')
                ax2.plot(range(len(current_data), len(results)), 
                         future_data['heart_rate'] / 2,  # Scale heart rate for visibility
                         linestyle='--', color='red')
            
            # Plot boundary line between actual and predicted days
            plt.axvline(x=len(current_data)-0.5, color='gray', linestyle='--')
        
        plt.title('Vital Signs')
        plt.xlabel('Time Points')
        plt.ylabel('Temperature (°C)')
        ax2.set_ylabel('SpO2 (%) / Heart Rate (bpm, scaled)')
        plt.grid(True, alpha=0.3)
        
        # Create a combined legend for both axes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("models/visualizations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"models/visualizations/prediction_results_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Prediction visualization saved to {plot_path}")

def main():
    """Main function for making predictions."""
    parser = argparse.ArgumentParser(description="Make predictions with the health prediction model")
    parser.add_argument("--input", required=True, help="Path to the input data CSV file")
    parser.add_argument("--model_path", help="Path to the saved model")
    parser.add_argument("--days_ahead", type=int, default=7, help="Number of days to predict ahead")
    parser.add_argument("--output", help="Path to save the prediction results CSV")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    output_path = args.output or f"data/predictions/prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Making predictions using data from {args.input}")
    predictions = predict_health_status(
        args.input, 
        args.model_path, 
        args.days_ahead, 
        output_path, 
        not args.no_visualize
    )
    
    # Print summary of predictions
    n_issues = sum(predictions['predicted_class'] == 1)
    print(f"\nPrediction summary:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted health issues: {n_issues} ({n_issues/len(predictions)*100:.1f}%)")
    
    if 'is_future_prediction' in predictions.columns:
        future_pred = predictions[predictions['is_future_prediction']]
        if not future_pred.empty:
            n_future_issues = sum(future_pred['predicted_class'] == 1)
            print(f"Future health issues: {n_future_issues} out of {len(future_pred)} days")

if __name__ == "__main__":
    main() 