"""
Script to run the Health Prediction application locally.
"""

import os
import sys
from app import app

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set Flask environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Check command line arguments for host and port
    host = '127.0.0.1'  # localhost by default
    port = 5000  # default port
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print(f"Starting server at http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    
    # Run the Flask application
    app.run(host=host, port=port, debug=True) 