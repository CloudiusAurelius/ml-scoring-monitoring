"""
# 02_training/training.py

This script trains a Logistic Regression model on the data stored in /01_data/ingestdata/finaldata.csv.
It reads the configuration from a JSON file, processes the data, trains the model, and
saves the trained model to a specified output path.

Input parameters are provided via command line arguments.
    - config_file: Path to the configuration file containing input and output folder paths.
    - output_modelname: Name of the output file where the trained model will be saved.
"""
import argparse
import logging

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_processing.model_data_prep import process_data
from datetime import datetime

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
    model=LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False
        )
    
    # Fit the logistic regression to the data
    model.fit(X, y)
    return model


def go(args):
    
    logger.info("Starting model training process")

    # Define input variables
    # ----------------------

    # Target variable for training
    label = 'exited'
    logger.info(f"Target variable for training: {label}")

    # Categorical features for training
    categorical_features = ['corporation']

    # Input filename for training
    inputfilename = 'finaldata.csv'
    logger.info(f"Input filename for training: {inputfilename}")
   
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

    # Load the dataset        
    # --------------------------------------
    input_file_path = os.path.join(
        project_root,'01_data',
        config['output_folder_path'],
        inputfilename
        )       
    
    logger.info(f"Loading dataset from: {input_file_path}")
    df = load_dataset(input_file_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")    


    # Store training information
    # --------------------------------------
    logger.info("Storing training information")
    # Log the number of rows and columns in the training data.
    train_n_rows = df.shape[0]
    train_n_columns = df.shape[1]    
    logger.info(f"Training data:\
                number of rows: {train_n_rows},\
                number of columns: {train_n_columns}")    

    # Store feature names
    feature_names = df.drop([label], axis=1)\
        .columns\
        .tolist()
    logging.info(f"Feature names: {feature_names}")


    # Process the data
    # --------------------------------------
    logger.info("Processing data")
    # Split the dataset into features and target variable
    logger.info(f"Splitting dataset into features and target variable: {label}.\
                One-Hot Encoding categorical features: {categorical_features}")    
    X,y,encoder = process_data(
        df=df,
        label=label,
        categorical_features=categorical_features,
        training=True,
        encoder=None
    )

    logger.info(f"Processed data shapes: X: {X.shape}, y: {y.shape}")  
    logger.info(f"X preview: {X[:5]}")
    logger.info(f"y preview: {y[:5]}")
    
    # Train the model
    # --------------------------------------
    logger.info("Training the Logistic Regression model")
    model = train_model(X, y)
    logger.info("Training completed.")  
   

    # Save the trained model
    # --------------------------------------
    logger.info("Saving model to file")
    
    # Get the current date and time    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Current date and time: {current_datetime}")

    # Create a dictionary with model information.
    model_info = {
        "name": "logistic_regression_model",
        "created_at": current_datetime,
        "model": model,
        "params": model.get_params(),
        "features": feature_names,        
        "categorical_features": categorical_features,
        "label_column": label,
        "rows_train": train_n_rows,
        "columns_train": train_n_columns,
        "encoder": encoder   
    }
    logging.info(f"Saving model with model info: {model_info}")

    # Construct the model file path
    model_file_path = os.path.join(
        project_root,'02_training',
        config['output_model_path'],
        args.output_modelname
        )         

    # Save model with model info to a pickle file.
    with open(model_file_path, "wb") as filehandler:
        pickle.dump(model_info, filehandler)
    logger.info(f"Model trained and saved to {model_file_path}.")
    
    logger.info("-----Model training completed successfully.-----")


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
