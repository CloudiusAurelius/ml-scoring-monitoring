#!/bin/bash
export PYTHONPATH=$(pwd)

mlflow run . \
-P steps="data_ingestion"
-P steps="model_training"
-P steps="model_scoring"
-P steps="model_deployment" 
-P steps="diagnostics"
#-P steps="basic_cleaning"
#-P steps="data_check"
#-P steps="data_split"
#-P steps="train_random_forest"
#-P steps="test_regression_model"
