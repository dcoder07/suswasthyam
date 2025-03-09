# Deploying the Health Prediction System

This guide explains how to deploy the health prediction system with a backend server, making it accessible via web browser or API.

## Option 1: Local Deployment

For testing or personal use, you can deploy the system locally:

### Prerequisites

- Python 3.8+ installed
- Virtual environment (recommended)

### Steps

1. **Create a virtual environment (recommended)**:

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r web_requirements.txt
   ```

3. **Run the server**:

   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your web browser and navigate to http://localhost:5000

## Option 2: Cloud Deployment (Heroku)

For making the application accessible online:

### Prerequisites

- Git installed
- Heroku CLI installed
- Heroku account

### Steps

1. **Login to Heroku**:

   ```bash
   heroku login
   ```

2. **Create a new Heroku app**:

   ```bash
   heroku create your-health-prediction-app
   ```

3. **Create a Procfile**:
   Create a file named `Procfile` with the following content:

   ```
   web: gunicorn app:app
   ```

4. **Initialize Git and commit files**:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

5. **Deploy to Heroku**:

   ```bash
   git push heroku master
   ```

6. **Open the application**:
   ```bash
   heroku open
   ```

## Option 3: Docker Deployment

For containerized deployment:

### Prerequisites

- Docker installed

### Steps

1. **Create a Dockerfile**:
   Create a file named `Dockerfile` with the following content:

   ```Dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY . .

   RUN pip install --no-cache-dir -r web_requirements.txt

   EXPOSE 5000

   CMD ["python", "app.py"]
   ```

2. **Build the Docker image**:

   ```bash
   docker build -t health-prediction-system .
   ```

3. **Run the Docker container**:

   ```bash
   docker run -p 5000:5000 health-prediction-system
   ```

4. **Access the application**:
   Open your web browser and navigate to http://localhost:5000

## API Usage

The application exposes several API endpoints that can be used by other applications:

### 1. Predict Health Status

**Endpoint**: `/predict`  
**Method**: POST  
**Content-Type**: application/json

**Request body**:

```json
{
  "readings": [
    {
      "timestamp": "2023-01-01T08:00:00",
      "temperature": 37.5,
      "spo2": 95,
      "heart_rate": 88
    },
    {
      "timestamp": "2023-01-02T08:00:00",
      "temperature": 38.1,
      "spo2": 94,
      "heart_rate": 92
    },
    {
      "timestamp": "2023-01-03T08:00:00",
      "temperature": 38.5,
      "spo2": 92,
      "heart_rate": 95
    },
    {
      "timestamp": "2023-01-04T08:00:00",
      "temperature": 37.9,
      "spo2": 93,
      "heart_rate": 90
    }
  ]
}
```

**Response**:

```json
{
  "risk_score": 0.82,
  "prediction_class": 1,
  "prediction_source": "AI model",
  "risk_level": "High",
  "reading_assessments": [...],
  "malaria_risk": true
}
```

### 2. Batch Prediction

**Endpoint**: `/batch-predict`  
**Method**: POST  
**Content-Type**: multipart/form-data

**Form parameters**:

- `file`: CSV file with vital signs data
- `days_ahead` (optional): Number of days to predict ahead

### 3. Malaria Analysis

**Endpoint**: `/malaria-analysis`  
**Method**: GET

**Response**: Information about malaria symptoms and detection.

### 4. Model Status

**Endpoint**: `/model-status`  
**Method**: GET

**Response**: Status of the model and scaler loading.

## Customization

You can customize the application by:

1. **Modifying the UI**: Edit the HTML template in `templates/index.html`
2. **Adjusting prediction parameters**: Edit the configuration in `config.py`
3. **Adding authentication**: Implement user authentication for secure access

## Important Notes

- Always ensure you have regular backups of your model files
- For production deployment, set `debug=False` in the Flask app
- Consider implementing HTTPS for secure data transmission
- This application is for educational and informational purposes only, not for medical diagnosis
