"""
Minimal serverless function using Vercel's standard pattern.
"""
from http.server import BaseHTTPRequestHandler
import json
import datetime
import sys
import platform
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Basic response data
        response_data = {
            "status": "online",
            "message": "Suswasthyam API is running",
            "timestamp": datetime.datetime.now().isoformat(),
            "path": self.path,
            "version": "1.0.0-vercel"
        }
        
        # Add system info for health check endpoint
        if self.path == '/api/health':
            response_data.update({
                "python_version": sys.version,
                "platform": platform.platform(),
                "environment": "vercel",
                "vercel_env": {k: v for k, v in os.environ.items() if k.startswith(('VERCEL'))}
            })
        
        # Send the response as JSON
        self.wfile.write(json.dumps(response_data).encode()) 