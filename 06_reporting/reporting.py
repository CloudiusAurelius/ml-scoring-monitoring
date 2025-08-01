"""
# 06_reporting/reporting.py
This script generates a confusion matrix using the test data and the deployed model.
"""

import pickle
#from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
import logging


from data_processing.model_data_prep import process_data
from utils.common_utilities\
    import get_project_root,\
           load_config,\
           load_dataset
from diagnostics.diagnostics import model_predictions
           


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def confusion_matrix(df:pd.DataFrame, trained_model_filepath:str, config:dict, project_root:str) -> None:
    """
    Generate a confusion matrix using the test data and the deployed model.
    
    Inputs:
    - df: DataFrame containing the test data
    - trained_model_filepath: Path to the trained model file
    - project_root: Project root directory (required for constructing output file paths)
    - config: Configuration dictionary (required for locating output model path)
    
    Outputs:
    - None, but saves a confusion matrix plot to the output folder specified by project_root and config
    """

    # Make predictions
    # --------------------------------------
    logger.info("Making predictions on the test data and extracting the true labels")
    y_pred, y_true = model_predictions(df, trained_model_filepath)
    logger.info(f"Predictions made: {y_pred[:5]}")  # Log first 5 predictions
    logger.info(f"True labels: {y_true[:5]}")  # Log first 5 true labels


    # Confusion Matrix
    # --------------------------------------

    # Create confusion matrix using sklearn
    cm = sk_confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot
    output_plot_filepath = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        'confusion_matrix.png'
    )
    logger.info(f"Saving confusion matrix plot to {output_plot_filepath}")    
    plt.savefig(output_plot_filepath)


def go(args):
    """
    Main function to generate the confusion matrix.
    """
   # Define paths of source and destination
    # --------------------------------------    
    # Get project root
    project_root = get_project_root(logger)
    logger.info(f"Project root directory: {project_root}")

    # Load configuration
    config_filepath = os.path.join(project_root, args.config_file)
    config = load_config(config_filepath, logger)
    logger.info(f"Configuration loaded: {config}")

    # Load the dataset
    dataset_csv_path = os.path.join(
        project_root,'01_data',
        config['test_data_path'],
        args.input_data
        )           
    df = pd.read_csv(dataset_csv_path)
    logger.info(f"Dataset loaded from {dataset_csv_path} with shape {df.shape}")
    
    # Get the trained model path
    # --------------------------------------
    trained_model_filepath = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        args.input_modelinfo
    )
    
    # Generate confusion matrix
    # --------------------------------------
    confusion_matrix(df, trained_model_filepath, config, project_root)
    logger.info("-----Confusion matrix generated and saved successfully.-----") 




if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="Generate a confusion matrix using the test data and the deployed model.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

    parser.add_argument(
        "--input_data",
        type=str,
        help="Name of the input data file.",
        required=True
    )

    parser.add_argument(
        "--input_modelinfo",
        type=str,
        help="Name of the input model file.",
        required=True
    )

    args = parser.parse_args()    
    go(args)
