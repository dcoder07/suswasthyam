"""
Script to check health status prediction based on manual input of vital signs.
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_processor import DataProcessor
from model import HealthPredictionModel
import config
from utils.helpers import calculate_anomaly_score

def check_health_with_manual_input():
    """
    Take manual input of vital signs and predict health status.
    """
    print("\n========== HEALTH STATUS PREDICTION ==========")
    print("Please enter your vital signs for the past few days (at least 4 readings).")
    print("Malaria often presents with fever, low oxygen saturation, and elevated heart rate.")
    print("Normal ranges: Temperature: 36.5-37.5°C, SpO2: 95-100%, Heart Rate: 60-100 bpm\n")
    
    readings = []
    days = 0
    
    while days < 4:  # We need at least 4 readings for our model (based on TIME_STEPS)
        days += 1
        print(f"\nDay {days} vital signs:")
        
        try:
            temperature = float(input("Temperature (°C, e.g., 37.5): "))
            spo2 = int(input("Blood Oxygen/SpO2 (%, e.g., 97): "))
            heart_rate = int(input("Heart Rate (bpm, e.g., 75): "))
            
            # Validate input ranges
            if not (30 <= temperature <= 43):
                print("Warning: Temperature value seems unusual. Please double check.")
            if not (70 <= spo2 <= 100):
                print("Warning: SpO2 value should be between 70-100%.")
            if not (30 <= heart_rate <= 200):
                print("Warning: Heart Rate value seems unusual. Please double check.")
                
            readings.append({
                'timestamp': datetime.now() - timedelta(days=4-days),
                'temperature': temperature,
                'spo2': spo2,
                'heart_rate': heart_rate,
                'health_status': 0  # Default value, will be ignored for prediction
            })
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            days -= 1  # Try this day again
            
        if days >= 4:
            more = input("\nAdd more readings? (y/n): ").lower()
            if more == 'y':
                days += 1
    
    # Convert to DataFrame
    data = pd.DataFrame(readings)
    
    # Save data to a temporary file
    temp_file = "temp_input_data.csv"
    data.to_csv(temp_file, index=False)
    
    try:
        # Calculate anomaly scores directly since we can't guarantee model prediction
        anomaly_scores = [calculate_anomaly_score(r['temperature'], r['spo2'], r['heart_rate']) 
                           for r in readings]
        
        # Calculate an overall risk based on anomaly scores
        # If multiple days show high anomaly scores, risk is higher
        high_anomaly_days = sum(1 for score in anomaly_scores if score > 0.5)
        risk_score = max(anomaly_scores) * (1 + 0.1 * high_anomaly_days)
        risk_score = min(1.0, risk_score)  # Cap at 1.0
        
        # Try to use the model if possible
        model_prediction = None
        try:
            # Initialize model and load weights
            model = HealthPredictionModel()
            model_loaded = model.load()
            
            if model_loaded:
                # Initialize data processor
                data_processor = DataProcessor()
                scaler_loaded = data_processor.load_scaler()
                
                if scaler_loaded:
                    # Create a synthetic sequence if needed
                    if len(readings) >= config.TIME_STEPS:
                        # Use actual readings
                        processed_data = data_processor.preprocess_data(data, fit_scaler=False)
                        X, _ = data_processor.create_sequences(processed_data)
                        if len(X) > 0:
                            prediction_prob = float(model.predict(X[-1:]).flatten()[0])
                            model_prediction = prediction_prob
                
        except Exception as e:
            print(f"Note: Model prediction unavailable: {str(e)}")
        
        # Determine final risk assessment
        if model_prediction is not None:
            final_risk = model_prediction
            prediction_source = "AI model"
        else:
            final_risk = risk_score
            prediction_source = "anomaly calculation"
        
        prediction_class = int(final_risk > 0.5)
        
        # Display results
        print("\n========== PREDICTION RESULTS ==========")
        print(f"Health Status Risk: {final_risk:.2%} (based on {prediction_source})")
        
        if prediction_class == 1:
            print("WARNING: The prediction indicates an ABNORMAL health status.")
            print("This suggests potential health concerns that may require attention.")
            if final_risk > 0.8:
                print("HIGH RISK DETECTED - Consider seeking medical attention promptly.")
        else:
            print("The prediction indicates a NORMAL health status.")
            if final_risk > 0.3:
                print("However, there is some level of risk. Monitor your symptoms.")
        
        # Print most concerning readings
        print("\nVital Sign Analysis:")
        for i, reading in enumerate(readings):
            score = anomaly_scores[i]
            concern = "Normal"
            if score > 0.6:
                concern = "High Concern"
            elif score > 0.3:
                concern = "Moderate Concern"
            
            date_str = reading['timestamp'].strftime('%Y-%m-%d')
            print(f"Day {i+1} ({date_str}):")
            print(f"  Temperature: {reading['temperature']}°C, SpO2: {reading['spo2']}%, Heart Rate: {reading['heart_rate']} bpm")
            print(f"  Assessment: {concern} (Anomaly Score: {score:.2f})")
            
            # Flag malaria-consistent symptoms
            malaria_concerns = []
            if reading['temperature'] > 37.7:
                malaria_concerns.append("Fever")
            if reading['spo2'] < 95:
                malaria_concerns.append("Low oxygen")
            if reading['heart_rate'] > 100:
                malaria_concerns.append("Elevated heart rate")
                
            if malaria_concerns:
                print(f"  Note: {', '.join(malaria_concerns)} - can be consistent with malaria")
            print()
            
        # Malaria specific advice
        if prediction_class == 1 or any(score > 0.5 for score in anomaly_scores):
            print("\nRegarding Malaria Risk:")
            print("The abnormal vital signs pattern could be consistent with malaria, especially")
            print("if you've traveled to malaria-endemic regions recently. Malaria typically presents with:")
            print("  - Cyclical fever spikes")
            print("  - Chills and sweating")
            print("  - Headache and body aches")
            print("  - Fatigue")
            print("\nIMPORTANT: This is not a diagnostic tool for malaria. If you suspect malaria,")
            print("seek medical attention immediately for proper testing and diagnosis.")
        
        # Visualize the user's vital signs
        plt.figure(figsize=(12, 8))
        
        # Plot temperature
        plt.subplot(3, 1, 1)
        temps = [r['temperature'] for r in readings]
        dates = [r['timestamp'] for r in readings]
        plt.plot(dates, temps, 'ro-')
        plt.axhspan(36.5, 37.5, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('Temperature (°C)')
        plt.title('Your Temperature Readings')
        plt.grid(True, alpha=0.3)
        
        # Plot SpO2
        plt.subplot(3, 1, 2)
        spo2s = [r['spo2'] for r in readings]
        plt.plot(dates, spo2s, 'bo-')
        plt.axhspan(95, 100, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('SpO2 (%)')
        plt.title('Your Blood Oxygen Readings')
        plt.grid(True, alpha=0.3)
        
        # Plot heart rate
        plt.subplot(3, 1, 3)
        hrs = [r['heart_rate'] for r in readings]
        plt.plot(dates, hrs, 'go-')
        plt.axhspan(60, 100, color='green', alpha=0.2, label='Normal Range')
        plt.ylabel('Heart Rate (bpm)')
        plt.title('Your Heart Rate Readings')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('my_health_vitals.png')
        plt.close()
        
        print("\nA chart of your vital signs has been saved to 'my_health_vitals.png'")
        print("\nDISCLAIMER: This tool provides a risk assessment based on vital signs only.")
        print("It is NOT a diagnosis. Always consult healthcare professionals for proper medical advice.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    check_health_with_manual_input() 