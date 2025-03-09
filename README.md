# Adaptive Health Prediction Model

This project implements an adaptive machine learning model that predicts health outcomes based on vital signs data (temperature, SpO2, and heart rate) over time intervals.

## Features

- **Self-improving model**: Trains on initial data and continuously fine-tunes as new data arrives
- **Time-series analysis**: Considers temporal patterns in vital signs
- **Multiple prediction formats**: Can predict future values or health status classifications
- **Evaluation metrics**: Tracks model performance over time

## Project Structure

- `data/`: Directory for storing training and test datasets
- `models/`: Directory for saved model states
- `utils/`: Helper functions for data processing and model evaluation
- `config.py`: Configuration parameters
- `data_processor.py`: Data processing pipeline
- `model.py`: Core ML model implementation
- `train.py`: Training and fine-tuning procedures
- `predict.py`: Interface for making predictions
- `evaluate.py`: Model evaluation utilities

## Getting Started

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Prepare your initial dataset in CSV format with columns for timestamp, temperature, SpO2, heart rate, and target variable

3. Run initial training:

   ```
   python train.py --data_path your_initial_data.csv
   ```

4. Make predictions:

   ```
   python predict.py --input your_test_data.csv
   ```

5. Fine-tune with new data:
   ```
   python train.py --data_path new_data.csv --fine_tune
   ```
