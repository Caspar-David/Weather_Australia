from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os
from docker.types import Mount
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
host_path = os.environ["HOST_PROJECT_PATH"]

# Retry configuration for tasks
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),  
}
# Define the DAG
with DAG(
    'weather_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    default_args=default_args,
) as dag:
    
    append_new_row = DockerOperator(
    task_id='append_new_row',
    image='weather_australia_preprocessing:latest',
    command='python src/data/append_new_row.py',
    mounts=[
        Mount(source=f"{host_path}/data/raw", target="/app/data/raw", type="bind"),
    ],
    auto_remove=True,
    working_dir='/app',
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
)

    preprocessing = DockerOperator(
        task_id='preprocessing',
        image='weather_australia_preprocessing:latest',
        command='python src/pipelines/run_preprocessing.py',
        mounts=[
            Mount(source=f"{host_path}/data/processed", target="/app/data/processed", type="bind"),
            Mount(source=f"{host_path}/data/raw", target="/app/data/raw", type="bind"),
        ],
        auto_remove=True,
        working_dir='/app',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
    )

    modelling = DockerOperator(
    task_id='modelling',
    image='weather_australia_modelling:latest',
    command='python src/pipelines/run_modelling.py',
    mounts=[
        Mount(source=f"{host_path}/data/processed", target="/app/data/processed", type="bind"),
        Mount(source=f"{host_path}/mlruns", target="/mlruns", type="bind"),
    ],
    auto_remove=True,
    working_dir='/app',
    docker_url='unix://var/run/docker.sock',
    network_mode='weather_australia_default',
    mount_tmp_dir=False
)
    
    sync_dvc = DockerOperator(
        task_id='sync_dvc',
        image='weather_australia_modelling:latest',
        command='python src/models/sync_dvc.py',
        mounts=[
            Mount(source=f"{host_path}/data/processed", target="/app/data/processed", type="bind"),
            Mount(source=f"{host_path}/.dvc", target="/app/.dvc", type="bind"),
            Mount(source=f"{host_path}/.git", target="/app/.git", type="bind"),
            Mount(source=f"{host_path}/.gitignore", target="/app/.gitignore", type="bind"),
            Mount(source=f"{host_path}/.netrc", target="/root/.netrc", type="bind"),
            Mount(source=f"{host_path}", target="/app", type="bind"),  # for dvc.yaml, params.yaml etc.
        ],
        environment={
            "DVC_CONFIG_DIR": "/app/.dvc",
        },
        auto_remove=True,
        working_dir='/app',
        docker_url='unix://var/run/docker.sock',
        network_mode='weather_australia_default',
        mount_tmp_dir=False
    )

    append_new_row >> preprocessing >> modelling >> sync_dvc