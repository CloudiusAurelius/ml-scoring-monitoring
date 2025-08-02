# ML Scoring & Monitoring

Udacity Nanodegree: Machine Learning DevOps Engineer — Project 4

## Overview

This project implements an end-to-end ML pipeline with monitoring and automation. The workflow includes:

1. **Data Ingestion**
2. **Model Training, Scoring, and Deployment**
3. **Diagnostics**
4. **Reporting**
5. **Process Automation**

## Setup

- **Conda Environment:**  
    Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate ml_scmon
    ```

- **Configuration:**  
    - Hydra config: `config.yaml`  
        Enable debug mode:
        ```bash
        export HYDRA_FULL_ERROR=1
        ```
    - Main MLflow code: `main.py`

## Workflow

The training pipeline was constructed with MLFlow. For each step below the following two files are used:
- `conda.yml` to define the environment of the respective step
- `MLproject` to define the execution of the respective step

To run the complete training, deployment and reporting pipeline:
```bash
./run_pipeline.sh
```

**Common Scripts:**  
Reusable code for configuration, model loading, and data processing is in:
- `utils/` — configuration, model utilities
- `data processing/` — data preparation for training and inference
- `diagnostics/` - data, model and system monitoring (see below)

### Step 1: Data Ingestion

- Reads all CSVs from `/01_data/practicedata`, removes duplicates, and writes to `/01_data/ingestdata/finaldata.csv`.
- Scripts: `01_data/ingestion.py`
- Run:
    ```bash
    mlflow run . -P steps="data_ingestion"
    ```
- A record of the ingested files is stored in `/01_data/ingesteddata/ingestfiles.txt` 

### Step 2: Model Training

- Trains a Logistic Regression model on `/01_data/ingestdata/finaldata.csv`.
- Stores the model together with related information in `/models/trainedmodel.pkl`
- Scripts: `02_training/training.py`
- Run:
    ```bash
    mlflow run . -P steps="model_training"
    ```

### Step 3: Model Scoring

- Scores test data using the trained model.
- Scripts: `03_scoring/scoring.py`
- Run:
    ```bash
    mlflow run . -P steps="model_scoring"
    ```
- The result of the scoring is stored in file `latestscore.txt` in the respective model folder

### Step 4: Model Deployment

- Moves model files to the deployment folder.
- Scripts: `04_deployment/deployment.py`
- Run:
    ```bash
    mlflow run . -P steps="model_deployment"
    ```

## Diagnostics

- Scripts in `diagnostics/` check data integrity, model predictions, and system performance (e.g., missing values, execution time, dependencies).
- Scripts: `diagnostics/diagnostics.py`
- Run:
    ```bash
    mlflow run . -P steps="diagnostics"
    ```

## Reporting

- Scripts in `06_reporting/` generate model performance reports and statistics.
- Scripts: `06_reporting/reporting.py`, `06_reporting/app.py`, `06_reporting/apicalls.py`
- Run:
    ```bash
    mlflow run . -P steps="reporting"
    ```
- Output files are stored as `apireturns.txt` and `confusionmatrix.png` in the respective model folder

## API Usage

- The API (`app.py`) exposes endpoints for predictions and diagnostics.
- Start the server:
    ```bash
    python app.py
    ```
- Endpoints:
    - `/` - Default endpoint with welcome message
    - `/prediction` — Model predictions
    - `/scoring` — Scoring metrics
    - `/summarystats` — Summary statistics for the ingested data
    - `/diagnostics` — Model, Data, Systems diagnostics 
    

## Running the full process
The project root contains the script:
- `fullprocess.py`
It performs two checks:
- 1. check for new data in source folder
- 2. check for model drift
Depending on the result of the checks, a new training and deployment chain will be triggered.  

To automate this process and run it automatically in defined intervalls, a cronjob can be created. For example to run `fullprocess.py` every 10 minutes:
```Bash
crontab -e

*/10 * * * * /bin/bash /absolute/path/to/fullprocess.py >> /absolute/path/to/cron.log 2>&1
``` 




## Notes

- Activate the conda environment before running scripts.
- Enable Hydra full error messages for troubleshooting.
- See each directory for detailed scripts and documentation.

