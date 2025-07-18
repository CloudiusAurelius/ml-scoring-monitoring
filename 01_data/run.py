#!/usr/bin/env python
"""
Ingests data from input files and writes to an output file.
"""
import argparse
import logging
import wandb


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
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
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
    """
    if df.empty:
        logger.error("The DataFrame is empty. Exiting.")
        return
    
    logger.info(f"Saving DataFrame to {outputfilepath}")
    df.to_csv(outputfilepath, index=False)



def go(args):

    # Ingest data from the input folder, remove duplicates and save it to the output folder
    logger.info("Starting data ingestion process")
        
   

    # Load the configuration file
    logger.info("Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)


    # Load input and output folder paths from the configuration
    input_folder_path = config.get('input_folder_path', args.input_folder_path)
    output_folder_path = config.get('output_folder_path', args.output_folder_path)
    
    # Load all CSV files from the input folder and merge them into a single DataFrame
    logger.info(f"Loading data from input folder: {input_folder_path}")
    df = load_csv(input_folder_path)

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
    

    # Save the cleaned dataset to a new artifact    
    logger.info(f"Saving cleaned dataset to {outputfilepath}")
    outputfilepath = os.path.join(output_folder_path, args.output_filename)

    logger.info("Data ingestion completed successfully.")


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
