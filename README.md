# Weather Australia MLOps Pipeline

A reproducible, containerized MLOps pipeline for weather prediction in Australia using Docker Compose and Airflow.

---

## 🚀 Quick Start

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd Weather_Australia
```

---

### 1.1. Prepare the Data for Simulation

Before starting the pipeline, you **must run the data split script** to prepare the raw data for simulation:

```sh
python src/data/split_for_simulation.py
```

This will generate the necessary files in `data/raw/` for the pipeline to work correctly.

---

### 2. Set Up Your Local Path and Fernet Key and DVC + Git information

Create a `.env` file in the project root with the following content (replace the path with your absolute project path):

```
HOST_PROJECT_PATH=C:/datascientest/mlops/project/Weather_Australia
AIRFLOW__CORE__FERNET_KEY=YOUR_FERNET_KEY_HERE
```

**How to generate a Fernet key:**  
Run this command in your terminal and copy the output:

```sh
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the generated key as the value for `AIRFLOW__CORE__FERNET_KEY` in your `.env` file.

> **Note:**  
> The Fernet key is required for Airflow to encrypt/decrypt sensitive data.  
> For development/demo, you can use any generated key.  
> **Do not share this key publicly for production use.**

DVC needs a `.netrc` file. Create it in the root with this content: 
machine dagshub.com
  login your_username
  password your_private_token (https://dagshub.com/user/settings/tokens)


For DVC to work you also need your git credentials in the src/models/sync_dvc.py. Edit the entries on lines 13 and 14!

---

### 3. Initialize Airflow Database and Create Admin User

**Only needed on first setup or after wiping volumes:**

```sh
docker-compose run --rm airflow airflow db init
docker-compose run --rm airflow airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

---

### 4. Build and Start All Services

```sh
docker-compose up --build
```

- This will:
  - Run data preprocessing and save processed data to a shared volume.
  - Train the model and save it to the same volume.
  - Start MLflow for experiment tracking ([http://localhost:5000](http://localhost:5000)).
  - Launch the FastAPI service for predictions ([http://localhost:8000](http://localhost:8000)).
  - Start Airflow for orchestration ([http://localhost:8080](http://localhost:8080)).

**To stop all services:**

```sh
docker-compose down
```

---

### 5. Trigger the Pipeline

- Open [http://localhost:8080](http://localhost:8080) and log in with your Airflow admin credentials.
- Trigger the `weather_pipeline` DAG.

---

### 6. Test the API

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

### 7. Useful Notes

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **Airflow UI:** [http://localhost:8080](http://localhost:8080)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- All containers share data via mounted host folders (see `.env` and `docker-compose.yml`).
- If you change code or dependencies, re-run `docker-compose up --build`.
- The API will automatically load the model when it becomes available (no need to restart the API container).
- For a completely fresh start, use:
  ```sh
  docker-compose down -v
  ```
  and repeat steps 3–6.

---

**Enjoy your end-to-end, containerized MLOps workflow!**