"""
Training script for the health prediction model.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_processor import DataProcessor
from model import HealthPredictionModel
import config

def plot_training_history(history, output_path):
    """
    Plot and save training history.
    
    Args:
        history (dict): Training history
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_model(data_path, model_save_path=None, visualize=True):
    """
    Train the health prediction model on the initial dataset.
    
    Args:
        data_path (str): Path to the CSV file with training data
        model_save_path (str): Path to save the trained model
        visualize (bool): Whether to visualize training history
        
    Returns:
        tuple: (model, metrics) - trained model and evaluation metrics
    """
    # Initialize data processor and prepare data
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_data(data_path)
    
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Initialize and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = HealthPredictionModel(input_shape)
    
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    print("\nModel evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    if model_save_path:
        model.save(model_save_path)
    else:
        model.save()
    
    # Visualize training history
    if visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models/visualizations", exist_ok=True)
        plot_path = f"models/visualizations/training_history_{timestamp}.png"
        plot_training_history(history, plot_path)
        print(f"Training visualization saved to {plot_path}")
    
    return model, metrics

def fine_tune_model(data_path, model_save_path=None, visualize=True):
    """
    Fine-tune an existing model with new data.
    
    Args:
        data_path (str): Path to the CSV file with new data
        model_save_path (str): Path to the saved model to fine-tune
        visualize (bool): Whether to visualize fine-tuning history
        
    Returns:
        tuple: (model, metrics) - fine-tuned model and evaluation metrics
    """
    # Load the existing model
    model = HealthPredictionModel()
    if not model.load(model_save_path):
        raise FileNotFoundError(f"No model found at {model_save_path or config.MODEL_SAVE_PATH}")
    
    # Initialize data processor and prepare data
    data_processor = DataProcessor()
    data_processor.load_scaler()  # Load the existing scaler
    
    # Prepare new data for fine-tuning
    X_new, y_new = data_processor.prepare_new_data_for_fine_tuning(data_path)
    
    print(f"Fine-tuning data shape: {X_new.shape}")
    
    # Fine-tune the model
    history = model.fine_tune(X_new, y_new)
    
    # Evaluate fine-tuned model if we have labels
    if config.TARGET in data_processor.load_data(data_path).columns:
        X_test, y_test = X_new[-int(len(X_new) * 0.2):], y_new[-int(len(y_new) * 0.2):]  # Use last 20% as test set
        metrics = model.evaluate(X_test, y_test)
        
        print("\nFine-tuned model evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        metrics = None
        print("No labels in fine-tuning data, skipping evaluation.")
    
    # Visualize fine-tuning history
    if visualize and 'accuracy' in history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models/visualizations", exist_ok=True)
        plot_path = f"models/visualizations/fine_tuning_history_{timestamp}.png"
        
        # Simplify history format for plotting
        plot_history = {
            'loss': history['loss'],
            'val_loss': history.get('val_loss', []),
            'accuracy': history['accuracy'],
            'val_accuracy': history.get('val_accuracy', [])
        }
        
        plot_training_history(plot_history, plot_path)
        print(f"Fine-tuning visualization saved to {plot_path}")
    
    return model, metrics

def main():
    """Main function for model training or fine-tuning."""
    parser = argparse.ArgumentParser(description="Train or fine-tune the health prediction model")
    parser.add_argument("--data_path", required=True, help="Path to the data CSV file")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune an existing model")
    parser.add_argument("--model_path", help="Path to save or load the model")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    if args.fine_tune:
        print(f"Fine-tuning model with data from {args.data_path}")
        fine_tune_model(args.data_path, args.model_path, not args.no_visualize)
    else:
        print(f"Training new model with data from {args.data_path}")
        train_model(args.data_path, args.model_path, not args.no_visualize)

if __name__ == "__main__":
    main() 