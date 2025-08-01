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
import argparse

import logging
import subprocess

from data_processing.model_data_prep import process_data
from utils.common_utilities\
    import get_project_root,\
           load_config,\
           load_dataset,\
           load_model


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



##################Function to get model predictions
def model_predictions(df: pd.DataFrame, model_file_path: str) -> list:
    """
    Load the model and make predictions on the provided DataFrame.
    Inputs:
    - df: DataFrame containing the data to score
    - model_file_path: Path to the trained model file
    Outputs:
    - y_pred: List of predictions made by the model
    - y_test: List of true labels (if available)
    """
    
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

    logger.info(f"Processed data shapes: X: {X_test.shape}")  
    logger.info(f"X_test preview: {X_test[:5]}")
    logger.info(f"y_test preview: {y_test[:5]}")
    


    # Score the data with the loaded model
    # --------------------------------------
    logger.info("Scoring the model on the test data")
    
    # Prediction using the model    
    y_pred = model.predict(X_test)
    logger.info(f"Predictions made on the test data: {y_pred[:5]}")

    # Check that length of predictions matches length of test data
    if len(y_pred) != len(X_test):
        logger.error("Length of predictions does not match length of test data.")
        return []

    return y_pred.tolist(), y_test.tolist()  # Convert to list for consistency

##################Function to get summary statistics
def dataframe_summary(df: pd.DataFrame) -> list:
    """
    Calculate summary statistics of the dataset:
    - mean
    - median
    - standard deviation
    Inputs:
    - df: DataFrame containing the data to summarize
    Outputs:
    - summary_stats: List containing the summary statistics of every column
    """

    logger.info("Calculating summary statistics for the DataFrame")
    summary_stats = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mean = df[column].mean()
            median = df[column].median()
            std_dev = df[column].std()
            summary_stats.append({
                "column": column,
                "mean": mean,
                "median": median,
                "std_dev": std_dev
            })
        else:
            logger.warning(f"Column {column} is not numeric. Skipping summary statistics.")

    return summary_stats #return value should be a list containing all summary statistics

##################Function to calcuate percent of missing values
def missing_values_percent(df: pd.DataFrame) -> list:
    """
    Calculate the percentage of missing values in each column of the DataFrame.
    Inputs:
    - df: DataFrame containing the data to check for missing values
    Outputs:
    - missing_values: List containing the percentage of missing values for each column
    """

    logger.info("Calculating percentage of missing values in each column")
    missing_values = []
    total_rows = len(df)
    
    for column in df.columns:
        if total_rows > 0:            
            percent_missing = df[column].isna().sum() / total_rows * 100
            missing_values.append(percent_missing)
        else:
            logger.warning(f"DataFrame is empty. Cannot calculate missing values for column {column}.")

    return missing_values #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time() -> list:
    """
    Measure the execution time of the ingestion and training scripts.
    This function runs the ingestion.py and training.py scripts and returns their execution times.
    Inputs:
    - project_root: Path to the project root directory (default: None, will use current working directory)
    Outputs:
    - timings: List containing the execution times of ingestion and training scripts in seconds
    """
    logger.info("Measuring execution time of ingestion and training scripts")

    project_root = get_project_root(logger)
    ingestion_path = os.path.join(project_root, '01_data', 'ingestion.py')
    logger.info(f"Ingestion script path: {ingestion_path}")
    training_path = os.path.join(project_root, '02_training', 'training.py')    
    logger.info(f"Training script path: {training_path}")

    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()    
    os.system(f"python {ingestion_path}")
    ingestion_time = timeit.default_timer() - start_time
    logger.info(f"Ingestion script executed in {ingestion_time} seconds")

    training_time = timeit.default_timer()    
    os.system(f"python {training_path}")
    training_time = timeit.default_timer() - start_time
    logger.info(f"Training script executed in {training_time} seconds")

    return [ingestion_time, training_time]  #return a list of 2 timing values in seconds
    
##################Function to check dependencies
def check_outdated_packages() -> pd.DataFrame:
    """
    Returns a table which contains currently installed packages
    together with the latest available versions.
    Inputs:
    - None
    Outputs:
    - df: pandas Dataframe containing the collected information
    """    

    # Get current installed packages
    output = subprocess.check_output(
        ["pip", "list", "--format=json"],
        text=True
    )
    packages = json.loads(output)

    # Get latest version and store all information in a list
    packages_list = []
    for pkg in packages:
        name = pkg['name']
        current_version = pkg['version']
        
        try:
            # Get latest version using pip index
            result = subprocess.check_output(
                f"pip index versions {name}",
                shell=True,
                text=True
            )
            import re
            match = re.search(r"Available versions: ([\d\.]+)", result)
            if match:
                latest_version = match.group(1)
            else:
                # fallback: try to find the first version-like string
                version_match = re.search(r"\b\d+(\.\d+)+\b", result)
                latest_version = version_match.group(0) if version_match else "unknown"
        except Exception as e:
            latest_version = "unknown"

        packages_list.append({
            'name': name,
            'current': current_version,
            'latest': latest_version
        })

    # Create DataFrame
    df = pd.DataFrame(packages_list)

    # Optional: rename columns for clarity
    df.columns = ['name', 'current_version', 'latest_version']

    return df




def go(args):
    
    logger.info("Starting diagnostics")   
    

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
   

    # Load the dataset        
    # --------------------------------------
    input_file_path = os.path.join(
        project_root,'01_data',
        config['output_folder_path'],
        args.target_data
        )       
    
    logger.info(f"Loading dataset from: {input_file_path}")
    df = load_dataset(input_file_path, logger)
    logger.info(f"Dataset loaded with shape: {df.shape}") 

    # Model predictions
    # --------------------------------------  
    model_filepath = os.path.join(prod_deployment_path, args.input_modelinfo)
    y_pred,_ = model_predictions(df, model_filepath)
    logger.info(f"Model predictions: {y_pred[:5]}")  # Log first 5 predictions

    # Dataframe summary
    # --------------------------------------  
    summary_stats = dataframe_summary(df)
    logger.info(f"Summary statistics: {summary_stats}")

    # Missing values
    # --------------------------------------
    missing_values = missing_values_percent(df)
    logger.info(f"Missing values percentage: {missing_values}")

    # Execution time
    # --------------------------------------      
    timings = execution_time()
    logger.info(f"Execution times (seconds): Ingestion: {timings[0]}, Training: {timings[1]}")

    # Outdated packages
    # --------------------------------------  
    outdated_packages = check_outdated_packages()
    logger.info(f"Outdated packages check: {outdated_packages}")


    # Report message
    # --------------------------------------
    logger.info("Diagnostics completed.")

 



if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="Deploy the trained model and related files.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

    parser.add_argument(
        "--target_data",
        type=str,
        help="Name of the trained model file to be loaded.",
        required=True
    )
     
    parser.add_argument(
        "--input_modelinfo",
        type=str,
        help="Name of the trained model file to be loaded.",
        required=True
    )
       
    args = parser.parse_args()
    
    go(args)






    
