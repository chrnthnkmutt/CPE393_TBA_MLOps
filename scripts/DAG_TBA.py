from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import pandas as pd
import joblib
import sys
import os

from train import (feature_engineering, Model_Training, GetBaseModel, BasedLine2, ScoreDataFrame, save_model, load_model)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    feature_engineering = PythonOperator(
        task_id = 'feature_engineering',
        python_callable = feature_engineering
    )

    Model_Training = PythonOperator(
        task_id = 'Model_Training',
        python_callable = Model_Training
    )

    GetBaseModel = PythonOperator(
        task_id = 'GetBaseModel',
        python_callable = GetBaseModel
    )

    BasedLine2 = PythonOperator(
        task_id = 'BasedLine2',
        python_callable = BasedLine2
    )

    ScoreDataFrame = PythonOperator(
        task_id = 'ScoreDataFrame',
        python_callable = ScoreDataFrame
    )

    save_model = PythonOperator(
        task_id = 'save_model',
        python_callable = save_model
    )

    load_model = PythonOperator(
        task_id  = 'load_model',
        python_callable = load_model
    )

    end = EmptyOperator(task_id = 'end')

    start >> feature_engineering >> Model_Training >> GetBaseModel >> BasedLine2 >> ScoreDataFrame >> save_model >> load_model >> end