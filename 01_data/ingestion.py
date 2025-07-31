#!/usr/bin/env python
"""
# 01_data/ingestion.py

Ingests data from input files and writes to an output file.
Duplicates are removed.

Input parameters are provided via command line arguments.
    - config_file: Path to the configuration file containing input and output folder paths.
    - output_filename: Name of the output file where the final data will be saved.
"""
import argparse
import logging


import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

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


def load_csv(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a folder and merge them into a single DataFrame.
    Inputs:
    - folder_path: Path to the folder containing CSV files
    Outputs:
    - df: Merged DataFrame containing all data from the CSV files
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not all_files:
        logger.error("No CSV files found in the input folder. Exiting.")
        return pd.DataFrame()
    
    df_list = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    # Concatenate all DataFrames into one
    df = pd.concat(df_list, ignore_index=True)
    
    return df, all_files


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    Inputs:
    - df: DataFrame from which duplicates need to be removed
    Outputs:
    - df: DataFrame with duplicates removed
    """
    if df.empty:
        logger.error("The DataFrame is empty. Exiting.")
        return df
    
    logger.info(f"DataFrame shape before removing duplicates: {df.shape}")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"DataFrame shape after removing duplicates: {df.shape}")
    
    return df

def save_dataframe(df: pd.DataFrame, outputfilepath: str) -> None:
    """
    Save a DataFrame to a CSV file.
    Inputs:
    - df: DataFrame to be saved
    - outputfilepath: Path where the DataFrame will be saved
    Outputs:
    - None
    """
    if df.empty:
        logger.error("The DataFrame is empty. Exiting.")
        return
    
    logger.info(f"Saving DataFrame to {outputfilepath}")
    df.to_csv(outputfilepath, index=False)



def go(args):

    # Ingest data from the input folder, remove duplicates and save it to the output folder
    logger.info("Starting data ingestion process")        
   

    # Get the current working directory and project root
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    project_root = os.path.dirname(cwd)
    logger.info(f"Project root directory: {project_root}")
    
    # Load the configuration file
    config_filepath = os.path.join(project_root, args.config_file)
    if not os.path.exists(config_filepath):
        logger.error(f"\n***Configuration file {config_filepath} does not exist. Exiting.\n")
        return
    logger.info(f"Loading configuration from: {config_filepath}")    
    config = load_config(config_filepath)
    logger.info(f"Configuration loaded: {config}")


    # Load input and output folder paths from the configuration    
    input_folder_path = os.path.join(
                           project_root,
                           '01_data',
                           config['input_folder_path']
                        )
    logger.info(f"Input folder path: {input_folder_path}")

    output_folder_path = os.path.join(
                            project_root,
                            '01_data',
                            config['output_folder_path']
                        )
    logger.info(f"Output folder path: {output_folder_path}")
    
    # Load all CSV files from the input folder and merge them into a single DataFrame
    # Store the filenames in the input folder
    logger.info(f"Loading data from input folder: {input_folder_path}")
    df, all_files = load_csv(input_folder_path)

    # Check if the DataFrame is empty after merging
    if df.empty:
        logger.error("The merged DataFrame is empty. Exiting.")
        return

    # Log the shape of the merged DataFrame
    logger.info(f"Merged DataFrame shape after loading: {df.shape}")

    # Remove duplicates
    df = remove_duplicates(df)

    # Log the shape of the DataFrame after removing duplicates
    logger.info(f"Merged DataFrame shape after removing duplicates: {df.shape}")
    
    # Save the merged DataFrame to a CSV file
    outputfilepath = os.path.join(output_folder_path, args.output_filename)
    logger.info(f"Saving merged DataFrame to {outputfilepath}")
    df.to_csv(outputfilepath, index=False)
    

    # Save a record of the ingested filenames
    record_file_path = os.path.join(output_folder_path, 'ingested_files.txt')
    with open(record_file_path, 'w') as f:
        [f.write(f"{file}\n") for file in all_files]
        
    logger.info(f"Record of ingested files saved to {record_file_path}")

    logger.info("-----Data ingestion completed successfully.-----")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to the configuration file containing input and output folder paths.",
        required=True
    )

    parser.add_argument(
        "--output_filename", 
        type=str,
        help="Name of the output file where the final data will be saved.",
        required=True
    )
    
    args = parser.parse_args()

    go(args)
