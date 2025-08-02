#!/bin/bash
export PYTHONPATH=$(pwd)

echo "Running ML pipeline with MLflow..."

# Run the pipeline with individual steps
echo -e "\nRunning data ingestion step..."
mlflow run . \
-P steps="data_ingestion"

echo -e "\nRunning model training step..."
mlflow run . \
-P steps="model_training"

echo -e "\nRunning model scoring step..."
mlflow run . \
-P steps="model_scoring"

echo -e "\nRunning model deployment step..."
mlflow run . \
-P steps="model_deployment"

echo -e "\nRunning diagnostics step..."
mlflow run . \
-P steps="diagnostics"

echo -e "\nRunning reporting step..."
mlflow run . \
-P steps="reporting"


# Uncomment the following line to run the pipeline with all steps
# Run the pipeline with all steps
#mlflow run . \
#-P steps="data_ingestion,model_training,model_scoring,model_deployment
