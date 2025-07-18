#!/bin/bash
mlflow run. \
#- P steps="basic_cleaning"
#- P steps="data_check"
#- P steps="data_split"
#- P steps="train_random_forest"
#- P steps="test_regression_model"
