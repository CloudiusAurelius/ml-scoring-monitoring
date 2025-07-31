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
    - to enable debug with full Hydra errors:
    ```Bash
    export HYDRA_FULL_ERROR=1
    ```
- MLFlow main code in:
```main.py```


## 2. Workflow

The entire pipeline can be executed with:
```Bash
./run_pipeline.sh
```

### Step 1: Data Ingestion
- all csv files from folder ```/01_data/practicedata``` are read and written to:
    - ```/01_data/ingestdata/finaldata.csv``` (duplicate records are removed)

- all relevant scripts are stored in ```01_data```
    - ```conda.yml``` to define the relevant environment
    - ```MLproject``` to define the execution and input parameters
    - ```ingestion.py``` to run the operations

- **execute** step with:
```Bash
mlflow run . -P steps="data_ingestion"
```

### Step 2: Model Training
- a Logistic Regression model is trained on the data stored in /01_data/ingestdata/finaldata.csv. 

- all relevant scripts are stored in ```02_training```
    - ```conda.yml``` to define the relevant environment
    - ```MLproject``` to define the execution and input parameters
    - ```training.py``` to run the operations


- **execute** step with:
```Bash
mlflow run . -P steps="model_training"
```
