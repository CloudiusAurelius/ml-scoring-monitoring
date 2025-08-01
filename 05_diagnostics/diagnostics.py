"""
# 05_diagnostics/diagnostics.py

This script performs diagnostics on the trained model:
- calculates model predictions
- computes summary statistics of the dataset
- checks for missing values
- measures execution time of training and ingestion scripts
- checks for outdated packages
"""


import pandas as pd
import numpy as np
import timeit
import os
import json

import logging
from utils.common_utilities import get_project_root, load_config




logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    return #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 




def go(args):
    
    logger.info("Starting deployment process")   
    

    # Define paths of source and destination
    # --------------------------------------    
    # Get project root
    project_root = get_project_root(logger)
    logger.info(f"Project root directory: {project_root}")

    # Load configuration
    config_filepath = os.path.join(project_root, args.config_file)
    config = load_config(config_filepath, logger)
    logger.info(f"Configuration loaded: {config}")
    
    # Get the deployment path
    prod_deployment_path = os.path.join(
        project_root,
        '04_deployment',
        config['prod_deployment_path']        
    )
    logger.info(f"Deployment path: {prod_deployment_path}")
   

    # Model predictions
    # --------------------------------------  


    # Dataframe summary
    # --------------------------------------  


    # Execution time
    # --------------------------------------  


    # Outdated packages
    # --------------------------------------  



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy the trained model and related files.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

       
    args = parser.parse_args()
    
    go(args)


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    return #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
