"""
# 03_scoring/scoring.py

This script:
- loads test data from the path specified in the config.json file,
- reads a trained model from a pickle file
- calculates the F1 score of the model on the test data
- writes the F1 score to a file named latestscore.txt in the output folder path specified in config.json.

Input parameters:
    - config_file: Path to the config.json file containing paths for dataset and model.
    - input_modelname: Name of the trained model file to be loaded.

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

from data_processing.model_data_prep import process_data
from utils.common_utilities\
    import get_project_root,\
           load_config,\
           load_dataset,\
           load_model

from sklearn.metrics\
    import fbeta_score,\
           precision_score,\
           recall_score,\
           roc_auc_score


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> tuple:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    roc_auc = roc_auc_score(y, preds)
    return precision, recall, fbeta, roc_auc



def save_metrics(metrics_dict: dict, output_score_filename: str):
    """
    Save the model metrics to a file.
    Inputs: 
    - metrics_dict: Dictionary containing model metrics.
    - output_score_filename: Path to the output file where metrics will be saved.
    Ouputs:
    - File containing the model metrics.
    """
    with open(output_score_filename, 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")


def go(args):
    
    logger.info("Starting model scoring process")

    # Define input variables
    # ----------------------
        
    # Get the project root
    project_root = get_project_root(logger)   
    logger.info(f"Project root directory: {project_root}")

    # Load the configuration file
    # --------------------------------------
    config_filepath = os.path.join(project_root, args.config_file)
    if not os.path.exists(config_filepath):
        logger.error(f"\n***Configuration file {config_filepath} does not exist. Exiting.\n")
        return
    logger.info(f"Loading configuration from: {config_filepath}")    
    config = load_config(config_filepath, logger)
    logger.info(f"Configuration loaded: {config}")


    # Load the dataset        
    # --------------------------------------
    input_file_path = os.path.join(
        project_root,'01_data',
        config['test_data_path'],
        args.input_data
        )       
    
    logger.info(f"Loading dataset from: {input_file_path}")
    df = load_dataset(input_file_path, logger)
    logger.info(f"Dataset loaded with shape: {df.shape}") 


    # Load the model and the encoder        
    # --------------------------------------    
    # Get the model file path
    model_file_path = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        args.input_modelinfo
    )
    logger.info(f"Loading model from: {model_file_path}")
    
    # Load the model and encoder
    model_name,\
    model_created_at,\
    model,\
    encoder,\
    label,\
    categorical_features = load_model(model_file_path, logger)
    logger.info(f"Model load complete: {model_name} created at: {model_created_at} ")
    

    # Process the data
    # --------------------------------------
    logger.info("Processing data")    
    logger.info(f"Splitting dataset into features and target variable: {label}.\
                One-Hot Encoding categorical features: {categorical_features}")    
    X_test,y_test,_ = process_data(
        df=df,
        label=label,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder
    )

    logger.info(f"Processed data shapes: X: {X_test.shape}, y: {y_test.shape}")  
    logger.info(f"X preview: {X_test[:5]}")
    logger.info(f"y preview: {y_test[:5]}")


    # Score the data with the loaded model
    # --------------------------------------
    logger.info("Scoring the model on the test data")
    
    # Prediction using the model    
    y_pred = model.predict(X_test)
    logger.info(f"Predictions made on the test data: {y_pred[:5]}")


    # Compute model metrics
    # ---------------------------------------   
    logger.info("Computing model metrics")
    precision,\
    recall,\
    fbeta,\
    roc_auc = compute_model_metrics(y_test, y_pred)

    logger.info(f"Model metrics:\
                Precision: {precision:.4f},\
                Recall: {recall:.4f},\
                F-beta: {fbeta:.4f},\
                ROC AUC: {roc_auc:.4f}")


    # Save the model metrics to a file
    # --------------------------------------    
    # Define the output file path
    output_score_filename = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        args.output_score_filename
    )
    logger.info(f"Saving F1 score to: {output_score_filename}")
    
    # Save the metrics to the file
    metrics_dict = {
        "model_name": model_name,
        "created_at": model_created_at,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
    save_metrics(metrics_dict, output_score_filename)
    logger.info(f"Model metrics saved to {output_score_filename}")


    logger.info("-----Model scoring completed successfully.-----")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scoring script for trained model")

    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

    parser.add_argument(
        "--input_data", 
        type=str,
        help="Name of the file which contains the data to be scored.",
        required=True
    )

    parser.add_argument(
        "--input_modelinfo", 
        type=str,
        help="Name of the trained model file to be loaded.",
        required=True
    )

    parser.add_argument(
        "--output_score_filename", 
        type=str,
        help="Name of the output file where the F1 score will be written.",
        default="latestscore.txt"
    )
   
    args = parser.parse_args()

    go(args)

