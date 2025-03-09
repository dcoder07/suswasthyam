"""
Vercel serverless function entry point.
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# This is required for Vercel serverless functions
# The app object will be used by the Vercel Python runtime 