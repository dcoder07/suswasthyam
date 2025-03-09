"""
Flask API server for the Health Prediction Model - Vercel Lite Version.
This version is optimized for deployment on Vercel with minimal dependencies.
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import json
import base64
from io import BytesIO
import uuid
import sys
import platform
import datetime  # Use Python's standard datetime instead of pandas

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
    """Generate health status prediction."""
    readings = session.get('readings', [])
    
    if len(readings) < 4:
        return jsonify({'error': 'At least 4 readings are required for prediction.'}), 400
    
    try:
        # Simple anomaly detection without scikit-learn or pandas
        anomaly_scores = []
        
        for reading in readings:
            # Calculate anomaly score based on deviation from normal ranges
            temp_score = 0
            spo2_score = 0
            hr_score = 0
            
            # Temperature: normal range ~36.5-37.5°C
            temp = reading['temperature']
            if temp > 38.0:
                temp_score = min(1.0, (temp - 38.0) / 2.0 + 0.5)  # High fever
            elif temp > 37.5:
                temp_score = min(1.0, (temp - 37.5) / 1.0 * 0.5)  # Mild fever
            elif temp < 36.0:
                temp_score = min(1.0, (36.0 - temp) / 1.0 * 0.5)  # Low temp
                
            # SpO2: normal > 95%
            spo2 = reading['spo2']
            if spo2 < 90:
                spo2_score = min(1.0, (90 - spo2) / 10.0 + 0.5)  # Severe
            elif spo2 < 95:
                spo2_score = min(1.0, (95 - spo2) / 5.0 * 0.5)  # Mild
                
            # Heart rate: normal ~60-100 bpm
            hr = reading['heart_rate']
            if hr > 120:
                hr_score = min(1.0, (hr - 120) / 50.0 + 0.5)  # High
            elif hr > 100:
                hr_score = min(1.0, (hr - 100) / 40.0 * 0.5)  # Mild high
            elif hr < 50:
                hr_score = min(1.0, (50 - hr) / 10.0 * 0.5)  # Low
                
            # Combined score with more weight on SpO2
            anomaly_score = max(temp_score, spo2_score * 1.5, hr_score)
            anomaly_scores.append(anomaly_score)
        
        # Calculate an overall risk based on anomaly scores
        max_anomaly = max(anomaly_scores)
        avg_anomaly = sum(anomaly_scores) / len(anomaly_scores)
        high_anomaly_count = sum(1 for score in anomaly_scores if score > 0.5)
        risk_score = max(max_anomaly, avg_anomaly * 1.5) * (1 + 0.1 * high_anomaly_count)
        risk_score = min(1.0, risk_score)  # Cap at 1.0
        
        prediction_class = int(risk_score > 0.5)
        
        # Create analysis for each reading
        reading_analysis = []
        for i, reading in enumerate(readings):
            score = anomaly_scores[i]
            concern = "Normal"
            if score > 0.6:
                concern = "High Concern"
            elif score > 0.3:
                concern = "Moderate Concern"
            
            # Flag malaria-consistent symptoms
            malaria_concerns = []
            if reading['temperature'] > 37.7:
                malaria_concerns.append("Fever")
            if reading['spo2'] < 95:
                malaria_concerns.append("Low oxygen")
            if reading['heart_rate'] > 100:
                malaria_concerns.append("Elevated heart rate")
            
            reading_analysis.append({
                'reading': {
                    'temperature': reading['temperature'],
                    'spo2': reading['spo2'],
                    'heart_rate': reading['heart_rate']
                },
                'anomaly_score': score,
                'concern_level': concern,
                'malaria_concerns': malaria_concerns
            })
        
        # Simplified visualization data (just return data points, not actual plot)
        visualization_data = {
            'temperatures': [r['temperature'] for r in readings],
            'spo2s': [r['spo2'] for r in readings],
            'heart_rates': [r['heart_rate'] for r in readings]
        }
        
        # Build response
        response = {
            'risk_score': risk_score,
            'risk_percentage': f"{risk_score*100:.1f}%",
            'prediction_class': prediction_class,
            'prediction_source': "simplified anomaly detection",
            'anomaly_scores': anomaly_scores,
            'reading_analysis': reading_analysis,
            'visualization_data': visualization_data,
            'show_malaria_warning': prediction_class == 1 or any(score > 0.5 for score in anomaly_scores)
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
        'deployment_type': 'vercel-lite'
    }
    return jsonify(system_info)

# For Vercel serverless deployment
def handler(event, context):
    return app(event, context)

# Create necessary directories
for directory in ['static', 'templates']:
    os.makedirs(directory, exist_ok=True)

# This block only runs when directly executing this file
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 