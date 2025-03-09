"""
Evaluation module for the health prediction model.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve
)

from data_processor import DataProcessor
from model import HealthPredictionModel
import config

def evaluate_model(data_path, model_path=None, output_dir=None, visualize=True):
    """
    Evaluate the health prediction model on test data.
    
    Args:
        data_path (str): Path to the CSV file with test data
        model_path (str): Path to the saved model
        output_dir (str): Directory to save evaluation results
        visualize (bool): Whether to visualize evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Initialize model and load weights
    model = HealthPredictionModel()
    if not model.load(model_path):
        raise FileNotFoundError(f"No model found at {model_path or config.MODEL_SAVE_PATH}")
    
    # Initialize data processor and prepare data
    data_processor = DataProcessor()
    data_processor.load_scaler()
    
    # Load and preprocess data
    data = data_processor.load_data(data_path)
    processed_data = data_processor.preprocess_data(data, fit_scaler=False)
    
    # Check if target variable exists
    if config.TARGET not in processed_data.columns:
        raise ValueError(f"Target column '{config.TARGET}' not found in the data")
    
    # Create sequences
    X, y_true = data_processor.create_sequences(processed_data)
    
    if len(X) == 0:
        raise ValueError("Not enough data points for evaluation")
    
    # Make predictions
    y_pred_prob = model.predict(X).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_prob)
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Generate classification report
    cr = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = cr
    
    # Print metrics
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Save metrics to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
        
        # Remove non-serializable parts for JSON
        metrics_json = metrics.copy()
        metrics_json.pop("classification_report", None)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        print(f"\nEvaluation metrics saved to {metrics_path}")
    
    # Visualize results
    if visualize:
        visualize_evaluation(y_true, y_pred, y_pred_prob, output_dir)
    
    return metrics

def visualize_evaluation(y_true, y_pred, y_pred_prob, output_dir=None):
    """
    Visualize evaluation results.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_prob (np.ndarray): Predicted probabilities
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    if output_dir:
        vis_dir = os.path.join(output_dir, "visualizations")
    else:
        vis_dir = "models/visualizations"
    
    os.makedirs(vis_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.subplot(2, 2, 2)
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.subplot(2, 2, 3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot prediction distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_pred_prob, bins=20, alpha=0.7, color='purple')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(vis_dir, f"evaluation_results_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Evaluation visualization saved to {plot_path}")

def compare_models(data_paths, model_paths, model_names=None, output_dir=None):
    """
    Compare multiple model versions on the same test data.
    
    Args:
        data_paths (list): List of paths to test data files
        model_paths (list): List of paths to model files
        model_names (list): List of model names for labeling
        output_dir (str): Directory to save comparison results
        
    Returns:
        pd.DataFrame: Comparison results
    """
    if not model_names:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    all_metrics = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\nEvaluating {model_names[i]}...")
        
        # Get metrics for each data path
        model_metrics = []
        for data_path in data_paths:
            try:
                metrics = evaluate_model(data_path, model_path, output_dir, visualize=False)
                metrics['data_path'] = os.path.basename(data_path)
                model_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating {model_names[i]} on {data_path}: {str(e)}")
                continue
        
        # Calculate average metrics across all data paths
        if model_metrics:
            avg_metrics = {
                'model_name': model_names[i],
                'accuracy': np.mean([m['accuracy'] for m in model_metrics]),
                'precision': np.mean([m['precision'] for m in model_metrics]),
                'recall': np.mean([m['recall'] for m in model_metrics]),
                'f1_score': np.mean([m['f1_score'] for m in model_metrics]),
                'roc_auc': np.mean([m['roc_auc'] for m in model_metrics])
            }
            all_metrics.append(avg_metrics)
    
    # Create comparison DataFrame
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        
        # Save comparison results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
            comparison_df.to_csv(comparison_path, index=False)
            
            print(f"\nModel comparison saved to {comparison_path}")
            
            # Create comparison visualization
            visualize_model_comparison(comparison_df, output_dir)
        
        return comparison_df
    
    return None

def visualize_model_comparison(comparison_df, output_dir=None):
    """
    Visualize model comparison results.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame with comparison results
        output_dir (str): Directory to save visualization
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = comparison_df['model_name'].tolist()
    
    plt.figure(figsize=(12, 8))
    
    # Bar chart for each metric
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric.capitalize())
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * (len(metrics) - 1) / 2, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of each bar
    for i, metric in enumerate(metrics):
        for j, value in enumerate(comparison_df[metric]):
            plt.text(j + i * width, value + 0.01, f'{value:.3f}', 
                     ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    # Save visualization
    if output_dir:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, "visualizations", f"model_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Model comparison visualization saved to {plot_path}")

def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the health prediction model")
    parser.add_argument("--data_path", required=True, help="Path to the test data CSV file")
    parser.add_argument("--model_path", help="Path to the saved model")
    parser.add_argument("--output_dir", help="Directory to save evaluation results")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = f"models/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Evaluating model using data from {args.data_path}")
    evaluate_model(args.data_path, args.model_path, args.output_dir, not args.no_visualize)

if __name__ == "__main__":
    main() 