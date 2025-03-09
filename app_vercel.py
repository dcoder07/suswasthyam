"""
Ultra-minimal Flask API server for the Health Prediction Model - Vercel Lite Version.
This version uses almost no dependencies to stay well under Vercel's size limits.
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import json
import datetime
import uuid
import sys
import platform
import math

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))  # For session management
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/')
def index():
    """Render the main page."""
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['readings'] = []
    
    return render_template('index.html')

@app.route('/health')
def health_page():
    """Render the health check page."""
    return render_template('health.html')

@app.route('/debug')
def debug_page():
    """Render the debug page for local development."""
    return render_template('debug.html')

@app.route('/add_reading', methods=['POST'])
def add_reading():
    """Add a vital signs reading."""
    try:
        data = request.json
        
        # Validate input
        temperature = float(data.get('temperature'))
        spo2 = int(data.get('spo2'))
        heart_rate = int(data.get('heart_rate'))
        
        if not (30 <= temperature <= 43):
            return jsonify({'error': 'Temperature value seems unusual.'}), 400
        if not (70 <= spo2 <= 100):
            return jsonify({'error': 'SpO2 value should be between 70-100%.'}), 400
        if not (30 <= heart_rate <= 200):
            return jsonify({'error': 'Heart rate value seems unusual.'}), 400
        
        # Create reading with timestamp
        reading = {
            'timestamp': datetime.datetime.now().isoformat(),
            'temperature': temperature,
            'spo2': spo2,
            'heart_rate': heart_rate
        }
        
        # Add to session
        if 'readings' not in session:
            session['readings'] = []
        
        readings = session['readings']
        readings.append(reading)
        session['readings'] = readings
        
        return jsonify({'success': True, 'reading_count': len(readings)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_readings', methods=['GET'])
def get_readings():
    """Get all readings for the current session."""
    readings = session.get('readings', [])
    return jsonify(readings)

@app.route('/clear_readings', methods=['POST'])
def clear_readings():
    """Clear all readings."""
    session['readings'] = []
    return jsonify({'success': True})

@app.route('/predict', methods=['POST'])
def predict():
    """Generate basic health status prediction with minimal processing."""
    readings = session.get('readings', [])
    
    if len(readings) < 1:
        return jsonify({'error': 'At least 1 reading is required for prediction.'}), 400
    
    try:
        # Very simplified risk assessment
        temp_risks = []
        spo2_risks = []
        hr_risks = []
        
        for reading in readings:
            # Temperature risk (0-1 scale)
            temp = reading['temperature']
            if temp > 38.0:
                temp_risks.append(0.8)  # High fever
            elif temp > 37.5:
                temp_risks.append(0.5)  # Mild fever
            elif temp < 36.0:
                temp_risks.append(0.4)  # Low temperature
            else:
                temp_risks.append(0.1)  # Normal
                
            # SpO2 risk (0-1 scale)
            spo2 = reading['spo2']
            if spo2 < 90:
                spo2_risks.append(0.9)  # Severe
            elif spo2 < 95:
                spo2_risks.append(0.6)  # Moderate
            else:
                spo2_risks.append(0.1)  # Normal
                
            # Heart rate risk (0-1 scale)
            hr = reading['heart_rate']
            if hr > 120:
                hr_risks.append(0.7)  # High
            elif hr > 100:
                hr_risks.append(0.4)  # Elevated
            elif hr < 50:
                hr_risks.append(0.5)  # Low
            else:
                hr_risks.append(0.1)  # Normal
        
        # Calculate average risks
        avg_temp_risk = sum(temp_risks) / len(temp_risks) if temp_risks else 0
        avg_spo2_risk = sum(spo2_risks) / len(spo2_risks) if spo2_risks else 0
        avg_hr_risk = sum(hr_risks) / len(hr_risks) if hr_risks else 0
        
        # Weight SpO2 more heavily as it's a critical indicator
        weighted_risk = (avg_temp_risk * 0.3) + (avg_spo2_risk * 0.5) + (avg_hr_risk * 0.2)
        
        # Classify prediction
        prediction_class = 1 if weighted_risk > 0.5 else 0
        
        # Extract readings data for client-side visualization
        data_points = {
            'temperatures': [r['temperature'] for r in readings],
            'spo2s': [r['spo2'] for r in readings],
            'heart_rates': [r['heart_rate'] for r in readings]
        }
        
        # Build response
        response = {
            'risk_score': weighted_risk,
            'risk_percentage': f"{weighted_risk*100:.1f}%",
            'prediction_class': prediction_class,
            'prediction_source': "basic risk calculation",
            'visualization_data': data_points,
            'show_malaria_warning': prediction_class == 1
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Vercel."""
    # Gather system information for debugging
    system_info = {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('FLASK_', 'PYTHON', 'VERCEL'))},
        'is_vercel': 'VERCEL' in os.environ,
        'deployment_type': 'vercel-ultra-lite'
    }
    return jsonify(system_info)

# Handler for Vercel serverless function
def handler(event, context):
    """Vercel serverless function handler."""
    return app(event, context)

# This block only runs when directly executing this file
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 