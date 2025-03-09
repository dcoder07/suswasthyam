"""
Helper functions for the health prediction model.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tensorflow as tf
import random

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_synthetic_data(n_days=14, readings_per_day=4, 
                           normal_range={'temperature': (36.5, 37.2), 
                                         'spo2': (95, 100), 
                                         'heart_rate': (60, 100)},
                           abnormal_range={'temperature': (37.5, 39.5), 
                                          'spo2': (85, 94), 
                                          'heart_rate': (90, 130)},
                           abnormal_days=None,
                           noise_level=0.1,
                           output_path=None):
    """
    Generate synthetic vital signs data for testing.
    
    Args:
        n_days (int): Number of days to generate data for
        readings_per_day (int): Number of readings per day
        normal_range (dict): Range for normal vital signs
        abnormal_range (dict): Range for abnormal vital signs
        abnormal_days (list): List of days with abnormal readings
        noise_level (float): Level of random noise to add
        output_path (str): Path to save the generated data
        
    Returns:
        pd.DataFrame: Generated data
    """
    # Set default abnormal days if not provided
    if abnormal_days is None:
        # Randomly select 20% of days to be abnormal
        n_abnormal = max(1, int(n_days * 0.2))
        abnormal_days = sorted(random.sample(range(n_days), n_abnormal))
    
    # Create timestamp sequence
    start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    hours_interval = 24 // readings_per_day
    
    timestamps = []
    temperatures = []
    spo2_values = []
    heart_rates = []
    health_statuses = []
    
    for day in range(n_days):
        is_abnormal_day = day in abnormal_days
        
        for reading in range(readings_per_day):
            # Calculate timestamp
            current_timestamp = start_date + timedelta(days=day, hours=reading * hours_interval)
            timestamps.append(current_timestamp)
            
            # Determine vital sign ranges based on health status
            if is_abnormal_day:
                temp_range = abnormal_range['temperature']
                spo2_range = abnormal_range['spo2']
                hr_range = abnormal_range['heart_rate']
                health_status = 1
            else:
                temp_range = normal_range['temperature']
                spo2_range = normal_range['spo2']
                hr_range = normal_range['heart_rate']
                health_status = 0
            
            # Generate vital signs with some randomness
            temperature = np.random.uniform(temp_range[0], temp_range[1])
            spo2 = np.random.uniform(spo2_range[0], spo2_range[1])
            heart_rate = np.random.uniform(hr_range[0], hr_range[1])
            
            # Add some noise
            temperature += np.random.normal(0, noise_level * 0.5)
            spo2 += np.random.normal(0, noise_level * 2)
            heart_rate += np.random.normal(0, noise_level * 5)
            
            # Ensure values are within realistic limits
            spo2 = min(100, max(70, spo2))
            heart_rate = min(180, max(40, heart_rate))
            
            # Add to lists
            temperatures.append(temperature)
            spo2_values.append(int(spo2))
            heart_rates.append(int(heart_rate))
            health_statuses.append(health_status)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'spo2': spo2_values,
        'heart_rate': heart_rates,
        'health_status': health_statuses
    })
    
    # Save to CSV if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Synthetic data saved to {output_path}")
    
    return data

def plot_vital_signs(data, output_path=None, show_health_status=True):
    """
    Plot vital signs data.
    
    Args:
        data (pd.DataFrame): DataFrame with vital signs data
        output_path (str): Path to save the plot
        show_health_status (bool): Whether to show health status markers
    """
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sort by timestamp
    data = data.sort_values(by='timestamp')
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot temperature
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='tab:red')
    ax1.plot(data['timestamp'], data['temperature'], color='tab:red', marker='o', label='Temperature')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # Create second y-axis for SpO2
    ax2 = ax1.twinx()
    ax2.set_ylabel('SpO2 (%)', color='tab:blue')
    ax2.plot(data['timestamp'], data['spo2'], color='tab:blue', marker='s', label='SpO2')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Create third y-axis for heart rate
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Heart Rate (bpm)', color='tab:green')
    ax3.plot(data['timestamp'], data['heart_rate'], color='tab:green', marker='^', label='Heart Rate')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # Format x-axis
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    ax1.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    # Add health status markers if requested
    if show_health_status and 'health_status' in data.columns:
        # Add markers for abnormal health status
        abnormal_data = data[data['health_status'] == 1]
        if not abnormal_data.empty:
            ax1.scatter(abnormal_data['timestamp'], abnormal_data['temperature'], 
                       color='red', s=100, alpha=0.5, zorder=5, label='Abnormal Status')
    
    # Add a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    plt.title('Vital Signs Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Vital signs plot saved to {output_path}")
    else:
        plt.show()

def calculate_anomaly_score(temperature, spo2, heart_rate):
    """
    Calculate an anomaly score based on vital signs.
    
    Args:
        temperature (float): Body temperature in Celsius
        spo2 (int): Blood oxygen saturation in percentage
        heart_rate (int): Heart rate in beats per minute
        
    Returns:
        float: Anomaly score between 0 and 1
    """
    # Define normal ranges
    temp_normal = (36.5, 37.5)
    spo2_normal = (95, 100)
    hr_normal = (60, 100)
    
    # Calculate how far each value is from the normal range
    temp_score = max(0, (temperature - temp_normal[1]) / 2.0) if temperature > temp_normal[1] else \
                 max(0, (temp_normal[0] - temperature) / 2.0)
    
    spo2_score = max(0, (spo2_normal[0] - spo2) / 15.0)  # Lower SpO2 is worse
    
    hr_score = max(0, (heart_rate - hr_normal[1]) / 50.0) if heart_rate > hr_normal[1] else \
               max(0, (hr_normal[0] - heart_rate) / 30.0)
    
    # Combine scores with weights
    # Temperature and SpO2 are more critical indicators
    combined_score = 0.4 * temp_score + 0.4 * spo2_score + 0.2 * hr_score
    
    # Scale to 0-1 range
    return min(1.0, combined_score)

def track_model_performance(model_path, data_paths, metrics_path=None):
    """
    Track model performance over time with different datasets.
    
    Args:
        model_path (str): Path to the model
        data_paths (list): List of paths to test data files
        metrics_path (str): Path to save the performance metrics
        
    Returns:
        dict: Performance metrics over time
    """
    from model import HealthPredictionModel
    from data_processor import DataProcessor
    import config
    
    # Initialize model and data processor
    model = HealthPredictionModel()
    if not model.load(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    data_processor = DataProcessor()
    data_processor.load_scaler()
    
    # Track performance
    performance = {}
    
    for data_path in data_paths:
        try:
            # Prepare data
            data = data_processor.load_data(data_path)
            processed_data = data_processor.preprocess_data(data, fit_scaler=False)
            
            if config.TARGET not in processed_data.columns:
                print(f"Skipping {data_path}: No target column found")
                continue
                
            # Create sequences
            X, y_true = data_processor.create_sequences(processed_data)
            
            if len(X) == 0:
                print(f"Skipping {data_path}: Not enough data points")
                continue
                
            # Make predictions
            y_pred_prob = model.predict(X).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_pred_prob))
            }
            
            # Add to performance tracking
            dataset_name = os.path.basename(data_path)
            performance[dataset_name] = metrics
            
        except Exception as e:
            print(f"Error processing {data_path}: {str(e)}")
    
    # Save performance metrics if path provided
    if metrics_path and performance:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(performance, f, indent=4)
        print(f"Performance metrics saved to {metrics_path}")
    
    return performance 