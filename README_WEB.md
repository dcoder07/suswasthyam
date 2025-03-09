# Health Prediction System - Web Application

This is a web-based interface for the Adaptive Health Prediction Model that allows users to input vital signs data and receive predictions about potential health risks, including malaria.

## Features

- **User-friendly web interface**: Input temperature, SpO2, and heart rate readings through a simple form
- **Session-based data tracking**: Record multiple readings over time
- **Visual insights**: Automatically generates charts to visualize vital signs trends
- **Detailed analysis**: Provides comprehensive assessment of each reading
- **Malaria risk indicators**: Flags vital sign patterns that could be consistent with malaria
- **Responsive design**: Works on desktop and mobile devices

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**:

   ```
   git clone https://github.com/yourusername/health-prediction-system.git
   cd health-prediction-system
   ```

2. **Install dependencies and set up the application**:

   ```
   python setup.py --install-only
   ```

3. **Generate sample data (optional)**:

   ```
   python setup.py --generate-data
   ```

4. **Train the model** (if you don't have a pre-trained model):
   ```
   python train.py --data_path data/sample_dataset.csv
   ```

## Running the Web Application

1. **Start the server**:

   ```
   python setup.py --host 0.0.0.0 --port 5000
   ```

   For development mode:

   ```
   python setup.py --host 127.0.0.1 --port 5000 --debug
   ```

2. **Access the application**:
   Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Add vital signs readings**:

   - Enter temperature (°C), SpO2 (%), and heart rate (bpm) values
   - Click "Add Reading" to record the data
   - Add at least 4 readings for the prediction to work properly

2. **Generate prediction**:

   - After adding sufficient readings, click the "Get Prediction" button
   - The system will analyze your data and display the results

3. **Interpret the results**:
   - The risk assessment indicates the likelihood of health issues
   - Visual charts show your vital signs trends over time
   - The detailed analysis provides insights for each reading
   - If malaria patterns are detected, specific guidance will be shown

## Deployment Options

### Docker Deployment

1. **Build the Docker image**:

   ```
   docker build -t health-prediction-app .
   ```

2. **Run the container**:
   ```
   docker run -p 5000:5000 health-prediction-app
   ```

### Cloud Deployment

The application can be deployed to various cloud platforms:

- **Heroku**: Use the provided Procfile
- **AWS Elastic Beanstalk**: Follow AWS deployment guidelines
- **Google Cloud Run**: Package as a container and deploy

## Security Considerations

This web application stores user data in the server's session, which is suitable for development but may need enhancement for production:

- Add user authentication for longer-term data storage
- Implement HTTPS to encrypt data transmission
- Consider HIPAA compliance if using in healthcare settings

## Limitations

- This tool is for informational purposes only and is NOT a medical diagnostic device
- Always consult healthcare professionals for proper medical advice
- The model's predictions are based solely on vital signs data and may not capture all health conditions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras for machine learning capabilities
- Flask for the web framework
- Bootstrap for the responsive UI
