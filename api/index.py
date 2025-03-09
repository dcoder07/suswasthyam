"""
Minimal API endpoint for Vercel deployment.
"""
from flask import Flask, jsonify
from datetime import datetime
import os
import sys
import platform

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Catch-all route that returns API information"""
    return jsonify({
        "status": "online",
        "message": "Suswasthyam API is running",
        "path": path,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-micro"
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": "vercel"
    })

# Required for Vercel to route properly
def handler(event, context):
    return app(event, context) 