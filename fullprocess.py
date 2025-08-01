

#import training
#import scoring
#import deployment
#import diagnostics
#import reporting
import subprocess
import logging
import os

from utils.common_utilities\
    import get_project_root,\
           load_config
           


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Define variables
# --------------------------------------
config_file = 'config.json'
ingested_files = 'ingestedfiles.txt'
score_filename = 'latestscore.txt'
deployed_score_filename = 'deployedscore.txt'

# Define paths
# --------------------------------------    
# Get project root
project_root = get_project_root(logger)
logger.info(f"Project root directory: {project_root}")

# Load configuration
config_filepath = os.path.join(project_root, config_file)
config = load_config(config_filepath, logger)
logger.info(f"Configuration loaded: {config}")

# ingested files path       
ingested_file_path = os.path.join(
    project_root,'01_data',
    config['output_folder_path'],
    ingested_files
)       

# source data path
source_data_path = os.path.join(
    project_root,'01_data',
    config['input_folder_path']
)


##################Check and read new data
#first, read ingestedfiles.txt
with open(ingested_file_path, 'r') as f:
    ingested_files_list = f.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files_list = [f for f in os.listdir(source_data_path) if f.endswith('.csv')]

# Compared the two lists to find new files
result = [item in ingested_files for item in source_files_list]

if not all(result):
    print("New data found in the source folder.")
    new_data_detected = True
else:
    print("No new data found in the source folder.")
    exit(0)


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

if new_data_detected:
    print("New data detected, proceeding with the pipeline.")
    
    # Run the data ingestion step
    logging.info("Running data ingestion step on the new data.")
    subprocess.run([
        "mlflow", "run", ".", 
        "-P", "steps=data_ingestion"
    ])
    
    # Run the model training step
    logging.info("Running model training step on the ingested data.")
    subprocess.run([
        "mlflow", "run", ".", 
        "-P", "steps=model_training"
    ])


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

# retrieve path to the latest score file
score_filepath = os.path.join(
    project_root,
    '02_training',
    config['output_model_path'],
    score_filename
)

deployed_score_filepath = os.path.join(
    project_root,
    '02_training',
    config['output_model_path'],
    deployed_score_filename
)

# make a safety copy of the original score file
logging.info(f"Copying original score file to {deployed_score_filepath}")
os.system(f"cp {score_filepath} {deployed_score_filepath}")

# read the score of the deployed model
with open(deployed_score_filepath, 'r') as file:    
    for line in file:
        if line.startswith("fbeta:"):
            fbeta_value_deployed = float(line.split(":")[1].strip())
            break


# update score file with the newly trained model
logging.info("Re-scoring the data with the newly trained model.")
subprocess.run([
    "mlflow", "run", ".",
    "-P", "steps=data_ingestion", 
    "-P", "model_scoring"    
])

# read the score of the new model
with open(score_filepath, 'r') as file:    
    for line in file:
        if line.startswith("fbeta:"):
            fbeta_value_new = float(line.split(":")[1].strip())
            break



# read the scores from new score file
if fbeta_value_new != fbeta_value_deployed:
    model_drift_detected = True
    logging.info("Model drift detected.")
else:
    model_drift_detected = False
    logging.info("No model drift detected.")
    exit(0)  # Exit the script if no drift is detected


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if model_drift_detected:
    print("Model drift detected, proceeding with re-deployment.")


    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logging.info("Re-deploying the model due to detected drift.")
    subprocess.run([
        "mlflow", "run", ".", 
        "-P", "deployment",    
    ])

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model

    subprocess.run([
        "mlflow", "run", ".", 
        "-P", "diagnostics",
        "-P", "reporting"
    ])







