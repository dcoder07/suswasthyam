"""
Setup script for the Health Prediction System.
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'joblib',
        'tensorflow',
        'flask',
        'flask_cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing Python packages."""
    print("Installing missing dependencies...")
    
    for package in missing_packages:
        print(f"Installing {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error installing {package}: {result.stderr.decode()}")
            return False
    
    return True

def create_directories():
    """Create required directories."""
    directories = ['data', 'models', 'static', 'templates', 'utils']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_files():
    """Check if all required files exist."""
    required_files = [
        'app.py',
        'config.py',
        'data_processor.py',
        'model.py',
        'templates/index.html',
        'utils/helpers.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def run_application(host, port, debug):
    """Run the Flask application."""
    print(f"\nStarting the Health Prediction System on http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    os.environ['FLASK_APP'] = 'app.py'
    
    if debug:
        os.environ['FLASK_ENV'] = 'development'
        debug_flag = "--debug"
    else:
        debug_flag = ""
    
    subprocess.run([sys.executable, "-m", "flask", "run", 
                   "--host", host, "--port", str(port), debug_flag])

def generate_sample_data():
    """Generate sample data if data directory is empty."""
    data_dir = Path('data')
    sample_data_path = data_dir / 'sample_dataset.csv'
    
    # Check if data directory is empty
    if not any(data_dir.iterdir()):
        print("Generating sample dataset...")
        
        # Try to import necessary functions
        try:
            from utils.helpers import generate_synthetic_data
            
            # Generate data
            generate_synthetic_data(
                n_days=14, 
                readings_per_day=4,
                output_path=str(sample_data_path)
            )
            
            print(f"Sample dataset created at {sample_data_path}")
            
        except ImportError:
            print("Could not generate sample data. utils.helpers module not found.")
    else:
        print("Data directory already contains files. Skipping sample data generation.")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup and run the Health Prediction System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--install-only", action="store_true", help="Only install dependencies, don't run app")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    
    args = parser.parse_args()
    
    print("Health Prediction System Setup")
    print("==============================\n")
    
    # Check dependencies
    print("Checking Python dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        if not install_dependencies(missing_packages):
            print("Failed to install dependencies. Please install them manually.")
            sys.exit(1)
    else:
        print("All dependencies are installed.")
    
    # Create directories
    print("\nCreating required directories...")
    create_directories()
    
    # Check required files
    print("\nChecking required files...")
    missing_files = check_files()
    
    if missing_files:
        print(f"Warning: The following files are missing: {', '.join(missing_files)}")
        print("Some functionality may not work correctly.")
    else:
        print("All required files are present.")
    
    # Generate sample data if requested
    if args.generate_data:
        print("\nGenerating sample data...")
        generate_sample_data()
    
    # Exit if only installing
    if args.install_only:
        print("\nSetup completed successfully.")
        sys.exit(0)
    
    # Run the application
    try:
        run_application(args.host, args.port, args.debug)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    
if __name__ == "__main__":
    main() 