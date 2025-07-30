# ML-scoring-monitoring
Udacity Nanodegree Machine Learning DevOps Engineer - Project 4


## O. Overview
The project comprises the following 5 steps:
- 1. Data Ingestion
- 2. Training, Scoring, Deploying
- 3. Diagnostics
- 4. Reporting
- 5. Process Automation

## 1. Setup

- the conda environment was set up by:
    - creating the file _environment.yml_ and activating the environment with:
```Bash		
conda env create -f environment.yml	
conda activate ml_scmon
```


- Hydra Configuration file in :
```config.yaml```
- MLFlow main code in:
```main.py```


## 2. Workflow
### Step 1: Data Ingestion
- all csv files from folder /01_data/practicedata are read and written to:
- /01_data/ingestdata
- excute step with
```Bash
mlflow run . -P steps="data_ingestion"
```