# Hosting the Malaria Risk Prediction System

This document provides instructions for hosting the health prediction system in various environments.

## Local Deployment

### Option 1: Using the Deployment Script

The easiest way to run the application locally is to use the deployment script:

```bash
python deploy.py
```

This will set up the environment and start the server. A browser window will automatically open to the application at http://localhost:5000.

### Option 2: Running Directly with Python

You can also run the application directly:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create necessary directories:

   ```bash
   mkdir -p data/sample models static templates
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Access the application at http://localhost:5000

## Docker Deployment

You can deploy the application using Docker for better isolation and easier deployment:

1. Build the Docker image:

   ```bash
   docker build -t malaria-risk-prediction .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 malaria-risk-prediction
   ```

3. Access the application at http://localhost:5000

## Cloud Deployment

### Deploying to Azure App Service

1. Install the Azure CLI and log in:

   ```bash
   az login
   ```

2. Create a resource group if you don't have one:

   ```bash
   az group create --name myResourceGroup --location eastus
   ```

3. Create an App Service plan:

   ```bash
   az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux
   ```

4. Create a web app:

   ```bash
   az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name your-app-name --runtime "PYTHON:3.9"
   ```

5. Configure the web app to use Gunicorn:

   ```bash
   az webapp config set --resource-group myResourceGroup --name your-app-name --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:app"
   ```

6. Deploy the application:
   ```bash
   az webapp deployment source config-local-git --name your-app-name --resource-group myResourceGroup
   git remote add azure <git-url-from-previous-command>
   git push azure main
   ```

### Deploying to Google Cloud Run

1. Install the Google Cloud SDK and initialize:

   ```bash
   gcloud init
   ```

2. Build and push the Docker image to Google Container Registry:

   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/malaria-risk-prediction
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy malaria-risk-prediction --image gcr.io/your-project-id/malaria-risk-prediction --platform managed
   ```

## Important Considerations for Production

1. **Security**: For production, make sure to:

   - Set `debug=False` in the Flask application
   - Configure proper HTTPS
   - Add authentication if needed
   - Consider implementing rate limiting

2. **Database**: For storing user data and predictions:

   - Consider using a database like PostgreSQL or MongoDB
   - Update the application to save and retrieve data from the database

3. **Scaling**: For high traffic situations:

   - Use load balancing
   - Consider horizontal scaling options
   - Optimize the model for faster predictions

4. **Monitoring**:

   - Implement logging
   - Set up performance monitoring
   - Configure alerts for system issues

5. **Data Privacy**:
   - Ensure compliance with healthcare data regulations
   - Implement data encryption
   - Add disclaimers about data usage

## Troubleshooting

If you encounter issues with the deployment:

1. Check the logs:

   ```bash
   # For Docker
   docker logs <container-id>

   # For Azure
   az webapp log tail --name your-app-name --resource-group myResourceGroup

   # For Google Cloud Run
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=malaria-risk-prediction"
   ```

2. Verify the model files exist in the `models` directory

3. Ensure all dependencies are properly installed

4. Check that port 5000 is not already in use by another application
