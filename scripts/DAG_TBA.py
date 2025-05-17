from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import pandas as pd
import joblib
import sys

sys.path.append('/home/santitham/airflow/dags/CPE393_TBAOps')

from scripts.train_dag import feature_engineer, Model_Training, GetBasedModel, BasedLine2, ScoreDataFrame, save_model, load_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 5, 14),
    'retries': 1,
}

with DAG(
    'DAG_TBA',
    default_args = default_args,
    schedule = '0 0 * * *',
    catchup = False,
    tags = ['TBA_MLOps'],
) as dag:
    
    start = EmptyOperator(task_id = 'start')

    feature_engineer = PythonOperator(
        task_id = 'feature_engineer',
        python_callable = feature_engineer
    )

    Model_Training = PythonOperator(
        task_id = 'Model_Training',
        python_callable = Model_Training
    )

    GetBasedModel = PythonOperator(
        task_id = 'GetBasedModel',
        python_callable = GetBasedModel
    )

    BasedLine2 = PythonOperator(
        task_id = 'BasedLine2',
        python_callable = BasedLine2,
        op_kwargs = {
            'X_train': "{{ task_instance.xcom_pull(task_ids='Model_Training')['X_train_path'] }}",
            'y_train': "{{ task_instance.xcom_pull(task_ids='Model_Training')['y_train_path'] }}",
            'models': "{{ task_instance.xcom_pull(task_ids='GetBasedModel') }}"
        }
    )

    ScoreDataFrame = PythonOperator(
        task_id = 'ScoreDataFrame',
        python_callable = ScoreDataFrame,
        op_kwargs = {
            'names': "{{ task_instance.xcom_pull(task_ids='BasedLine2')['names'] }}",
            'results': "{{ task_instance.xcom_pull(task_ids='BasedLine2')['results'] }}"
        }
    )

    save_model = PythonOperator(
        task_id = 'save_model',
        python_callable = save_model,
        op_kwargs = {
            'model_type': "{{ task_instance.xcom_pull(task_ids='BasedLine2')['best_model']['name'] }}",
            'filename': "best_model_{{ ts_nodash }}.pkl"
        }
    )

    load_model = PythonOperator(
        task_id = 'load_model',
        python_callable = load_model,
        op_kwargs = {
            'filename': "{{ task_instance.xcom_pull(task_ids='save_model')['filename'] }}"
        }
    )

    end = EmptyOperator(task_id = 'end')

    start >> feature_engineer >> Model_Training >> GetBasedModel >> BasedLine2 >> ScoreDataFrame >> save_model >> load_model >> end