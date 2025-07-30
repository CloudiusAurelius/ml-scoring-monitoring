import argparse
import logging

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def load_config(config_file: str) -> dict:
    """
    Load configuration from a JSON file.
    Inputs:
    - config_file: Path to the configuration file
    Outputs:
    - config: Dictionary containing the configuration parameters
    """
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} does not exist. Exiting.")
        return {}
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def get_project_root() -> str:
    """
    Get the project root directory.
    Inputs:
    - None
    Outputs:
    - project_root: Path to the project root directory
    """
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    project_root = os.path.dirname(cwd)
    return project_root


def process_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Process the DataFrame to separate features and target variable.
    Inputs:
    - df: DataFrame containing the dataset
    - target: Name of the target variable column
    Outputs:
    - X: Features as a NumPy array
    - y: Target variable as a NumPy array
    """    
    X = df.drop(target, axis=1).values    
    y = df[target].values
    return X,y


def load_dataset(input_file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    Inputs:
    - input_file_path: Path to the input CSV file
    Outputs:
    - df: Loaded DataFrame
    """         
    if not os.path.exists(input_file_path):
        logger.error(f"Dataset file {input_file_path} does not exist. Exiting.")
        return pd.DataFrame()        
    df = pd.read_csv(input_file_path)        
    return df


def train_model(X, y) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    Inputs:
    - X: Features
    - y: Target variable
    Outputs:
    - model: Trained Logistic Regression model
    """
    
    # Create a Logistic Regression model
    model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # Fit the logistic regression to the data
    model.fit(X, y)
    return model
    



def go():
    
    logger.info("Starting model training process")

    # Define input variables
    target = 'exited'
    logger.info(f"Target variable for training: {target}")

    inputfilename = 'finaldata.csv'
    logger.info(f"Input filename for training: {inputfilename}")
   
    # Get the current working directory and project root    
    logger.info(f"Project root directory: {project_root}")
    project_root = get_project_root()   
    
    # Load the configuration file
    config_filepath = os.path.join(project_root, args.config_file)
    if not os.path.exists(config_filepath):
        logger.error(f"\n***Configuration file {config_filepath} does not exist. Exiting.\n")
        return
    logger.info(f"Loading configuration from: {config_filepath}")    
    config = load_config(config_filepath)
    logger.info(f"Configuration loaded: {config}")

    # Load the dataset        
    input_file_path = os.path.join(
        project_root,'01_data',
        config['output_folder_path'],
        inputfilename
        )       
    
    logger.info(f"Loading dataset from: {input_file_path}")
    df = load_dataset(input_file_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")    


    # Split the dataset into features and target variable
    logger.info(f"Splitting dataset into features and target variable: {target}")
    X,y = process_data(df, target)  
       
    
    # Train the model
    logger.info("Training the Logistic Regression model")
    model = train_model(X, y)
    logger.info("Model training completed successfully")
    

    # Save the trained model
    model_file_path = os.path.join(
        project_root,'02_training',
        config['output_model_path'],
        args.output_modelname
        )         
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)    
    logger.info(f"Model trained and saved to {model_file_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

    parser.add_argument(
        "--output_modelname", 
        type=str,
        help="Name of the output file where the final data will be saved.",
        required=True
    )
    

    args = parser.parse_args()

    go(args)
