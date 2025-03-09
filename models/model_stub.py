"""
Stub file to ensure the models directory exists in the repository.
The actual model files (*.h5, *.pkl) are excluded by .gitignore.
"""

import numpy as np

class ModelStub:
    """A stub model class that always returns a default value."""
    
    def predict(self, X):
        """Return a default prediction."""
        # Just return some reasonable default values
        batch_size = X.shape[0]
        return np.array([[0.3]] * batch_size)  # Low risk prediction

# This is here to make sure the models directory exists
# when deployed, even if the actual model files are not included 