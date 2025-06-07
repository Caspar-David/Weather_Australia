from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os
from docker.types import Mount

host_path = os.environ.get("HOST_PROJECT_PATH")

default_args = {
    'retries': 3,  # Number of retries before failing
    'retry_delay': timedelta(minutes=5),  # Wait 5 minutes between retries
}

with DAG(
    'weather_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    default_args=default_args,  # Add default_args here
) as dag:

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
            Mount(source=f"{host_path}/mlruns", target="/app/mlruns", type="bind"),
        ],
        auto_remove=True,
        working_dir='/app',
        docker_url='unix://var/run/docker.sock',
        network_mode='weather_australia_default',
        mount_tmp_dir=False
    )

    preprocessing >> modelling