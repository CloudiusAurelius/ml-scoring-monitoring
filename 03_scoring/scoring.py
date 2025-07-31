"""
# 03_scoring/scoring.py

This script:
- loads test data from the path specified in the config.json file,
- reads a trained model from a pickle file
- calculates the F1 score of the model on the test data
- writes the F1 score to a file named latestscore.txt in the output folder path specified in config.json.

Input parameters:
    - config_file: Path to the config.json file containing paths for dataset and model.

Output:
    - latestscore.txt: A file containing the F1 score of the model on the test
"""

import argparse
import logging

import pandas as pd
import numpy as np
import pickle
import os
import json

from utils.common_utilities import get_project_root, load_config

from sklearn import metrics



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def go(args):
    
    logger.info("Starting model training process")

    # Get the current working directory and project root
    project_root = get_project_root()   
    logger.info(f"Project root directory: {project_root}")

    # Load the configuration file
    # --------------------------------------
    config_filepath = os.path.join(project_root, args.config_file)
    if not os.path.exists(config_filepath):
        logger.error(f"\n***Configuration file {config_filepath} does not exist. Exiting.\n")
        return
    logger.info(f"Loading configuration from: {config_filepath}")    
    config = load_config(config_filepath)
    logger.info(f"Configuration loaded: {config}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scoring script for trained model")

    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )
   
    args = parser.parse_args()

    go(args)


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

