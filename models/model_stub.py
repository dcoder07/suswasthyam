"""
A stub model implementation for environments where TensorFlow is not available.
This helps the application run in deployment environments that might not support full TensorFlow.
"""

import numpy as np

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

# This is here to make sure the models directory exists
# when deployed, even if the actual model files are not included 