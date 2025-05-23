# MLOps_WeatherAUS_April2025

For learning purposes

---

## 🚀 Project Setup & Workflow

This guide explains how to set up, build, and run the full MLOps pipeline using Docker for the Weather Australia project.

---

### 1. Clone the Repository

Clone your repository and navigate into the project folder:

```sh
git clone <your-repo-url>
cd Weather_Australia
```

---

### 2. (Optional) Local Python Setup

If you want to run scripts or tests locally (outside Docker):

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Build Docker Images

Build each component as a separate Docker image:

```sh
docker build -f Dockerfile.preprocessing -t weather-preprocessing .
docker build -f Dockerfile.modelling -t weather-modelling .
docker build -f Dockerfile.api -t weather-api .
```

---

### 4. Run the Pipeline with Docker

All containers share the same Docker volume (`weather_data`) for data and model exchange.

**a. Preprocessing:**  
Runs data ingestion and transformation, outputs processed data to the shared Docker volume.

```sh
docker run --rm -v weather_data:/app/data/processed weather-preprocessing
```

**b. Modelling:**  
Trains the model using the processed data and saves the trained model to the same volume.

```sh
docker run --rm -v weather_data:/app/data/processed weather-modelling
```

**c. API:**  
Serves the trained model for predictions via FastAPI.

```sh
docker run --rm -p 8000:8000 -v weather_data:/app/data/processed weather-api
```

---

### 5. Test the API

You can test the API using the provided `tests/test_api.py` script:

```sh
python tests/test_api.py
```

Or, send a manual POST request using Python:

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    # ... your full feature dictionary here ...
}
response = requests.post(url, json=payload)
print(response.json())
```

---

### 6. Health Check

To check if the API is running:

- Visit [http://localhost:8000/health](http://localhost:8000/health) in your browser.
- You should see: `{"status": "ok"}`

---

### 7. Notes

- All containers share the same Docker volume (`weather_data`) for data and model exchange.
- You only need to rebuild Docker images if you change the code, requirements, or Dockerfiles.
- For development, you can use the virtual environment and run scripts locally as well.

---

**Enjoy your reproducible, containerized MLOps pipeline!**