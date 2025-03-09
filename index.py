"""
Simplified entry point for Vercel serverless deployment.
"""
from flask import Flask, jsonify

# Create the Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Basic home route that doesn't require templates"""
    return jsonify({
        "status": "online",
        "message": "Suswasthyam Health Prediction API is running",
        "version": "1.0.0-lite",
        "available_endpoints": [
            "/",
            "/api/health"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    import sys
    import platform
    import datetime
    import os
    
    # Gather basic system information
    system_info = {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('VERCEL', 'PYTHON'))},
        'deployment_type': 'vercel-micro'
    }
    return jsonify(system_info)

# Vercel serverless handler
def handler(event, context):
    """Handle Vercel serverless function request"""
    return app(event, context) 