"""
Deployment script for the health prediction web application.
This script sets up the necessary directories and starts the server.
"""

import os
import subprocess
import sys
import webbrowser
from time import sleep

def setup_environment():
    """Set up the necessary directories and environment."""
    print("Setting up environment...")
    
    # Create necessary directories
    os.makedirs('data/sample', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Check if requirements are installed
    try:
        import flask
        import pandas
        import numpy
        import tensorflow
        import matplotlib
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def start_server():
    """Start the Flask server."""
    print("\nStarting the health prediction server...")
    print("The web interface will be available at: http://localhost:5000")
    
    # Open browser after a short delay
    def open_browser():
        sleep(2)  # Wait for the server to start
        webbrowser.open('http://localhost:5000')
    
    # Start the browser in a separate thread
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask server
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

def main():
    """Main function to run the deployment script."""
    print("=" * 80)
    print("Health Prediction System Deployment")
    print("=" * 80)
    
    setup_environment()
    start_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer shutdown requested. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 