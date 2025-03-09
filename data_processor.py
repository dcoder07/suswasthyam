"""
Data processing module for health prediction model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import config

class DataProcessor:
    """Data processor for the health prediction model."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        # Initialize with default feature ranges for fallback scaling
        self.feature_means = {
            'temperature': 37.2,  # Normal body temperature average
            'spo2': 96.5,         # Normal SpO2 average
            'heart_rate': 80      # Normal heart rate average
        }
        self.feature_stds = {
            'temperature': 1.0,   # Standard deviation for temperature
            'spo2': 3.0,          # Standard deviation for SpO2
            'heart_rate': 15.0    # Standard deviation for heart rate
        }
    
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data[config.TIMESTAMP_COL] = pd.to_datetime(data[config.TIMESTAMP_COL])
        return data
    
    def preprocess_data(self, data, fit_scaler=False):
        """
        Preprocess the data for training or prediction.
        
        Args:
            data (pd.DataFrame): Input data
            fit_scaler (bool): Whether to fit the scaler on this data
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Warning: Found missing values:\n{missing_values[missing_values > 0]}")
            data = data.dropna()
            
        # Sort by timestamp
        data = data.sort_values(by=config.TIMESTAMP_COL)
        
        try:
            # Scale features
            if fit_scaler or not self.is_scaler_fitted:
                self.scaler.fit(data[config.FEATURES])
                self.is_scaler_fitted = True
                
                # Save the scaler
                try:
                    os.makedirs(os.path.dirname(config.SCALER_SAVE_PATH), exist_ok=True)
                    joblib.dump(self.scaler, config.SCALER_SAVE_PATH)
                except Exception as e:
                    print(f"Warning: Could not save scaler: {str(e)}")
            
            scaled_features = self.scaler.transform(data[config.FEATURES])
        except Exception as e:
            print(f"Warning: Error using scaler: {str(e)}. Falling back to manual scaling.")
            # Fallback to manual scaling using predefined means and stds
            scaled_features = np.zeros((len(data), len(config.FEATURES)))
            for i, feature in enumerate(config.FEATURES):
                values = data[feature].values
                mean = self.feature_means.get(feature, values.mean())
                std = self.feature_stds.get(feature, values.std() or 1.0)  # Avoid division by zero
                scaled_features[:, i] = (values - mean) / std
        
        scaled_data = pd.DataFrame(scaled_features, columns=config.FEATURES)
        
        # Add target variable
        if config.TARGET in data.columns:
            scaled_data[config.TARGET] = data[config.TARGET].values
            
        # Add timestamp for reference
        scaled_data[config.TIMESTAMP_COL] = data[config.TIMESTAMP_COL].values
        
        return scaled_data
    
    def load_scaler(self):
        """Load a saved scaler."""
        try:
            if os.path.exists(config.SCALER_SAVE_PATH):
                self.scaler = joblib.load(config.SCALER_SAVE_PATH)
                self.is_scaler_fitted = True
                return True
        except Exception as e:
            print(f"Warning: Could not load scaler: {str(e)}. Will use default scaling.")
        
        # If we get here, we couldn't load the scaler
        print("Using default feature scaling values")
        return False
    
    def create_sequences(self, data, time_steps=None):
        """
        Create sequences for time series prediction.
        
        Args:
            data (pd.DataFrame): Input data
            time_steps (int): Number of time steps in each sequence
            
        Returns:
            tuple: (X, y) where X is the sequence input and y is the target
        """
        if time_steps is None:
            time_steps = config.TIME_STEPS
            
        X, y = [], []
        target_col_idx = len(config.FEATURES)
        
        # Convert DataFrame to numpy array
        values = data[config.FEATURES + [config.TARGET]].values
        
        for i in range(len(values) - time_steps):
            X.append(values[i:i+time_steps, :target_col_idx])
            y.append(values[i+time_steps, target_col_idx])
            
        return np.array(X), np.array(y)
    
    def prepare_train_test_data(self, data_path, test_size=None):
        """
        Prepare training and test data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file
            test_size (float): Test set proportion
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = config.TRAIN_TEST_SPLIT
            
        # Load and preprocess data
        data = self.load_data(data_path)
        processed_data = self.preprocess_data(data, fit_scaler=True)
        
        # Create sequences
        X, y = self.create_sequences(processed_data)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_prediction_data(self, data_path, time_steps=None):
        """
        Prepare data for prediction.
        
        Args:
            data_path (str): Path to the CSV file
            time_steps (int): Number of time steps to consider
            
        Returns:
            tuple: (X, timestamps)
        """
        if time_steps is None:
            time_steps = config.TIME_STEPS
            
        # Load scaler if not already fitted
        if not self.is_scaler_fitted:
            self.load_scaler()
        
        # Load and preprocess data
        data = self.load_data(data_path)
        processed_data = self.preprocess_data(data, fit_scaler=False)
        
        # For prediction, we might not have the target column
        if config.TARGET not in processed_data.columns:
            # Add a dummy target column
            processed_data[config.TARGET] = 0
        
        # Create sequences
        X, _ = self.create_sequences(processed_data, time_steps)
        
        # Get timestamps for reference
        timestamps = processed_data[config.TIMESTAMP_COL].values[time_steps:]
        
        return X, timestamps
    
    def prepare_new_data_for_fine_tuning(self, data_path):
        """
        Prepare new data for fine-tuning the model.
        
        Args:
            data_path (str): Path to the new data CSV file
            
        Returns:
            tuple: (X, y)
        """
        # Load scaler
        if not self.is_scaler_fitted:
            self.load_scaler()
        
        # Load and preprocess data
        data = self.load_data(data_path)
        processed_data = self.preprocess_data(data, fit_scaler=False)
        
        # Create sequences
        X, y = self.create_sequences(processed_data)
        
        return X, y 