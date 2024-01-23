from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
# Ajoutez le répertoire contenant main.py à la variable sys.path
# pour que Python puisse trouver le module main
dag_folder = os.path.dirname(__file__)
sys.path.append(dag_folder)
import main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 18),
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'weather_prediction',
    default_args=default_args,
    description='A DAG to fetch weather data, process it and train prediction models',
    schedule_interval=timedelta(minutes=5),
)

def fetch_data(**kwargs):
    main.fetch_weather_data('c12572cd545541941c716007ad70807b', ['paris', 'london', 'washington'])

def transform_to_csv(**kwargs):
    main.transform_data_into_csv(n_files=None, filename='fulldata.csv')
    main.transform_data_into_csv(n_files=20, filename='data.csv')

def train_and_evaluate(**kwargs):
    X, y = main.prepare_data('/app/clean_data/fulldata.csv')
    model, score = main.select_and_train_model(X, y)
    main.save_model(model, '/app/clean_data/model.pckl')

fetch_data_task = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_data,
    provide_context=True,
    dag=dag,
)

transform_to_csv_task = PythonOperator(
    task_id='transform_data_into_csv',
    python_callable=transform_to_csv,
    provide_context=True,
    dag=dag,
)

train_and_evaluate_task = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate,
    provide_context=True,
    dag=dag,
)

fetch_data_task >> transform_to_csv_task >> train_and_evaluate_task
