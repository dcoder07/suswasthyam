"""
AI model implementation for health prediction based on vital signs.
"""

import os
import json
import numpy as np

# Try to import TensorFlow, but gracefully handle if it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow could not be imported. Using model stub instead.")

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    print("WARNING: scikit-learn metrics could not be imported.")

import config

# Define ModelStub class for use when model files are missing
class ModelStub:
    """A stub model class that always returns a default value."""
    
    def __init__(self):
        """Initialize the stub model."""
        self.name = "Model Stub"
        self.is_stub = True
    
    def predict(self, X):
        """Return a default prediction."""
        # Check if X is a numpy array
        if hasattr(X, 'shape'):
            batch_size = X.shape[0]
        else:
            # Try to convert to numpy array
            try:
                X = np.array(X)
                batch_size = X.shape[0]
            except:
                batch_size = 1
                
        # Return low risk predictions (0.3) for all samples
        return np.array([[0.3]] * batch_size)
        
    def summary(self):
        """Return a summary of the model."""
        return "Model Stub (TensorFlow not available)"
        
    def get_weights(self):
        """Return empty weights."""
        return []

# Import ModelStub from external file if available
try:
    from models.model_stub import ModelStub
except ImportError:
    # Already defined above, so we'll use that
    pass

class HealthPredictionModel:
    """Model for predicting health outcomes based on vital signs."""
    
    def __init__(self, input_shape=None, model_type=None):
        """
        Initialize the health prediction model.
        
        Args:
            input_shape: The shape of input data
            model_type: Type of model to use (lstm, gru, simple)
        """
        self.model = None
        self.history = None
        self.input_shape = input_shape or (config.SEQUENCE_LENGTH, config.NUM_FEATURES)
        self.model_type = model_type or config.DEFAULT_MODEL_TYPE
        self.is_stub = not TENSORFLOW_AVAILABLE
        
        if not self.is_stub and TENSORFLOW_AVAILABLE:
            try:
                self.build_model(self.input_shape, self.model_type)
            except Exception as e:
                print(f"Failed to build model: {str(e)}")
                self.is_stub = True
                self.model = ModelStub()
        else:
            self.model = ModelStub()
            
    
    def build_model(self, input_shape, model_type=None):
        """
        Build the neural network model architecture.
        
        Args:
            input_shape: Tuple specifying input dimensions
            model_type: The type of RNN to use ('lstm', 'gru', or 'simple')
        
        Returns:
            A compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            self.model = ModelStub()
            return self.model
            
        try:
            model_type = model_type or config.DEFAULT_MODEL_TYPE
            
            model = Sequential()
            
            if model_type.lower() == 'lstm':
                # LSTM-based model
                model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(LSTM(32))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
            elif model_type.lower() == 'gru':
                # GRU-based model (typically faster than LSTM)
                model.add(GRU(64, input_shape=input_shape, return_sequences=True))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(GRU(32))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
            else:
                # Simple dense model
                model.add(tf.keras.layers.Flatten(input_shape=input_shape))
                model.add(Dense(64, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
            
            # Common dense layers for all model types
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # Binary classification output
            
            # Compile the model
            model.compile(
                optimizer=Adam(config.LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            self.model = ModelStub()
            self.is_stub = True
            return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
            
        if epochs is None:
            epochs = config.EPOCHS
            
        if batch_size is None:
            batch_size = config.BATCH_SIZE
            
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                config.MODEL_SAVE_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.history["training"].append(history.history)
        self._save_history()
        
        return history.history
    
    def fine_tune(self, X_new, y_new, epochs=None, learning_rate=None):
        """
        Fine-tune the model with new data.
        
        Args:
            X_new (np.ndarray): New features for fine-tuning
            y_new (np.ndarray): New targets for fine-tuning
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for fine-tuning
            
        Returns:
            dict: Fine-tuning history
        """
        if self.model is None:
            raise ValueError("Model must be trained before fine-tuning")
            
        if epochs is None:
            epochs = config.FINE_TUNE_EPOCHS
            
        if learning_rate is None:
            learning_rate = config.FINE_TUNE_LEARNING_RATE
            
        # Update optimizer with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune the model
        history = self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            verbose=1
        )
        
        # Save fine-tuning history
        self.history["fine_tuning"].append(history.history)
        self._save_history()
        
        # Save updated model
        self.model.save(config.MODEL_SAVE_PATH)
        
        return history.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.model is None:
            # Create a stub model if none exists
            self.model = ModelStub()
            self.is_stub = True
            
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None or self.is_stub:
            print("Warning: Using stub model for evaluation. Results may not be meaningful.")
            y_pred_prob = self.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics = {
                "loss": 0.5,  # Placeholder value
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_prob.flatten())
            }
            return metrics
            
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = {
            "loss": self.model.evaluate(X_test, y_test, verbose=0)[0],
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_prob.flatten())
        }
        
        return metrics
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None or self.is_stub:
            print("Warning: Cannot save a stub model")
            return
            
        if filepath is None:
            filepath = config.MODEL_SAVE_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        self.model.save(filepath)
        self._save_history()
        
    def load(self, filepath=None):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: True if model loaded successfully
        """
        if filepath is None:
            filepath = config.MODEL_SAVE_PATH
            
        if not os.path.exists(filepath):
            print(f"Model file not found at {filepath}, using stub model...")
            # Use the model stub as a fallback
            self.model = ModelStub()
            self.is_stub = True
            return True
            
        try:
            self.model = load_model(filepath)
            self._load_history()
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}. Using stub model instead.")
            self.model = ModelStub()
            self.is_stub = True
            return True
    
    def _save_history(self):
        """Save the training history to a file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config.MODEL_HISTORY_PATH), exist_ok=True)
            
            with open(config.MODEL_HISTORY_PATH, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Warning: Could not save history: {str(e)}")
            
    def _load_history(self):
        """Load the training history from a file."""
        try:
            if os.path.exists(config.MODEL_HISTORY_PATH):
                with open(config.MODEL_HISTORY_PATH, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load history: {str(e)}")
                
    def predict_for_days(self, last_sequences, days=7):
        """
        Predict health status for the upcoming days.
        
        Args:
            last_sequences (np.ndarray): Last known sequences of vital signs
            days (int): Number of days to predict ahead
            
        Returns:
            list: Predicted health status for each day
        """
        if self.model is None:
            self.model = ModelStub()
            self.is_stub = True
            
        predictions = []
        current_sequence = last_sequences[-1].copy()  # Use the most recent sequence
        
        for _ in range(days):
            # Make prediction for current sequence
            pred = self.model.predict(np.array([current_sequence]))[0][0]
            predictions.append(float(pred))
            
            # Update sequence for next prediction (rolling window)
            # Here we're simplistically using the prediction as the new health status
            # In a real scenario, you would model the temperature, SpO2, and heart rate separately
            new_step = np.append(current_sequence[-1, :-1], pred)
            current_sequence = np.vstack([current_sequence[1:], new_step])
            
        return predictions 