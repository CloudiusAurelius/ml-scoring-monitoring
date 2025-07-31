"""
# utils/common_utilities.py

This module contains common utility functions used across different scripts.
"""

import logging
import json
import os

import pandas as pd
import pickle

def load_config(config_file: str, logger: logging.Logger) -> dict:
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



def get_project_root(logger: logging.Logger) -> str:
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


def load_dataset(input_file_path: str, logger: logging.Logger) -> pd.DataFrame:
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


def load_model(model_file_path: str, logger: logging.Logger):
    """
    Load a trained model from a file.
    Inputs:
    - model_file_path: Path to the model file
    Outputs:
    - model: Loaded model object
    - encoder: Loaded encoder object
    - label: Name of the label column
    - categorical_features: List of categorical features used in the model
    """
    if not os.path.exists(model_file_path):
        logger.error(f"Model file {model_file_path} does not exist. Exiting.")
        return None
    
    with open(model_file_path, 'rb') as filehandler:
       model_info = pickle.load(filehandler)
    model_name = model_info["name"]
    model_created_at = model_info["created_at"]
    model = model_info["model"]
    encoder = model_info["encoder"]
    label = model_info["label_column"]
    features = model_info["features"]
    categorical_features = model_info["categorical_features"]
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model created at: {model_created_at}")
    logger.info(f"Model loaded: {model}")
    logger.info(f"Encoder loaded: {encoder}")
    logger.info(f"Label column: {label}")
    logger.info(f"Features used: {features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    return model_name, model_created_at, model, encoder, label, categorical_features