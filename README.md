# Weather Australia MLOps Pipeline

A reproducible, containerized MLOps pipeline for weather prediction in Australia using Docker Compose.

---

## 🚀 Quick Start

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd Weather_Australia
```

---

### 2. Build and Run the Full Pipeline with Docker Compose

This project uses Docker Compose to orchestrate all steps: preprocessing, modelling, MLflow tracking, and serving the API.

**To build and start everything:**

```sh
docker compose up --build
```

- This will:
  - Run data preprocessing and save processed data to a shared Docker volume.
  - Train the model and save it to the same volume.
  - Start MLflow for experiment tracking (accessible at [http://localhost:5000](http://localhost:5000)).
  - Launch the FastAPI service for predictions (accessible at [http://localhost:8000](http://localhost:8000)).

**To stop all services:**

```sh
docker compose down
```

---

### 3. Test the API

After the pipeline is up and running, you can test the API in two ways:

#### a. Using the Provided Test Script

```sh
python tests/test_api.py
```

#### b. Manually with Python Requests

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    # ... your full feature dictionary here ...
}
response = requests.post(url, json=payload)
print(response.json())
```

#### c. Health Check

Visit [http://localhost:8000/health](http://localhost:8000/health) in your browser.  
You should see: `{"status": "ok"}`

---

### 4. Useful Notes

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- All containers share the `weather_data` Docker volume for data/model exchange.
- If you change code or dependencies, re-run `docker compose up --build`.
- For development, you can still use a Python virtual environment and run scripts locally.

---

**Enjoy your end-to-end, containerized MLOps workflow!**