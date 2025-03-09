"""
Flask API server for the Health Prediction Model.
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import uuid

from data_processor import DataProcessor
from model import HealthPredictionModel
import config
from utils.helpers import calculate_anomaly_score

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  # For session management
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global cache for model and data processor to avoid reloading
model_cache = None
data_processor_cache = None

def get_model():
    """Get or initialize the model."""
    global model_cache
    if model_cache is None:
        model_cache = HealthPredictionModel()
        model_cache.load()
    return model_cache

def get_data_processor():
    """Get or initialize the data processor."""
    global data_processor_cache
    if data_processor_cache is None:
        data_processor_cache = DataProcessor()
        data_processor_cache.load_scaler()
    return data_processor_cache

@app.route('/')
def index():
    """Render the main page."""
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['readings'] = []
    
    return render_template('index.html')

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
            'timestamp': datetime.now().isoformat(),
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
        # Convert readings to proper format
        formatted_readings = []
        for i, r in enumerate(readings):
            # Convert string timestamp back to datetime
            timestamp = datetime.fromisoformat(r['timestamp']) if isinstance(r['timestamp'], str) else r['timestamp']
            
            formatted_readings.append({
                'timestamp': timestamp,
                'temperature': float(r['temperature']),
                'spo2': int(r['spo2']),
                'heart_rate': int(r['heart_rate']),
                'health_status': 0  # Default value, ignored for prediction
            })
        
        # Calculate anomaly scores
        anomaly_scores = [calculate_anomaly_score(r['temperature'], r['spo2'], r['heart_rate']) 
                           for r in formatted_readings]
        
        # Calculate an overall risk based on anomaly scores
        high_anomaly_days = sum(1 for score in anomaly_scores if score > 0.5)
        risk_score = max(anomaly_scores) * (1 + 0.1 * high_anomaly_days)
        risk_score = min(1.0, risk_score)  # Cap at 1.0
        
        # Try to use the model if possible
        model_prediction = None
        try:
            model = get_model()
            data_processor = get_data_processor()
            
            # Create DataFrame from readings
            df = pd.DataFrame(formatted_readings)
            
            # Process data
            processed_data = data_processor.preprocess_data(df, fit_scaler=False)
            X, _ = data_processor.create_sequences(processed_data)
            
            if len(X) > 0:
                prediction_prob = float(model.predict(X[-1:]).flatten()[0])
                model_prediction = prediction_prob
                
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
        
        # Determine final risk assessment
        if model_prediction is not None:
            final_risk = model_prediction
            prediction_source = "AI model"
        else:
            final_risk = risk_score
            prediction_source = "anomaly calculation"
        
        prediction_class = int(final_risk > 0.5)
        
        # Create analysis for each reading
        reading_analysis = []
        for i, reading in enumerate(formatted_readings):
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
                'reading': reading,
                'anomaly_score': score,
                'concern_level': concern,
                'malaria_concerns': malaria_concerns
            })
        
        # Generate visualizations
        plt.figure(figsize=(10, 8))
        
        # Plot temperature
        plt.subplot(3, 1, 1)
        temps = [r['temperature'] for r in formatted_readings]
        dates = [i+1 for i in range(len(formatted_readings))]  # Use sequential numbers for x-axis
        plt.plot(dates, temps, 'ro-')
        plt.axhspan(36.5, 37.5, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Readings')
        plt.grid(True, alpha=0.3)
        
        # Plot SpO2
        plt.subplot(3, 1, 2)
        spo2s = [r['spo2'] for r in formatted_readings]
        plt.plot(dates, spo2s, 'bo-')
        plt.axhspan(95, 100, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('SpO2 (%)')
        plt.title('Blood Oxygen Readings')
        plt.grid(True, alpha=0.3)
        
        # Plot heart rate
        plt.subplot(3, 1, 3)
        hrs = [r['heart_rate'] for r in formatted_readings]
        plt.plot(dates, hrs, 'go-')
        plt.axhspan(60, 100, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('Heart Rate (bpm)')
        plt.title('Heart Rate Readings')
        plt.xlabel('Reading Number')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Build response
        response = {
            'risk_score': final_risk,
            'risk_percentage': f"{final_risk*100:.1f}%",
            'prediction_class': prediction_class,
            'prediction_source': prediction_source,
            'anomaly_scores': anomaly_scores,
            'reading_analysis': reading_analysis,
            'plot_image': plot_data,
            'show_malaria_warning': prediction_class == 1 or any(score > 0.5 for score in anomaly_scores)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Check if the model exists, otherwise warn
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("The application will use anomaly score calculation for predictions.")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 