"""
# 04_deployment/deployment.py

This script is responsible for deploying the trained model by copying:
- the record of the ingest files
- the latest model file
- the latest score file
into the production deployment directory.
"""

import logging
import os


from utils.common_utilities\
    import get_project_root,\
           load_config
           

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def copy_file(src: str, dest: str):
    """
    Copy a file from source to destination.
    Inputs:
    - src: Source file path
    - dest: Destination file path
    """
    if os.path.exists(src):
        logger.info(f"Copying {src} to {dest}")
        os.system(f"cp {src} {dest}")
        logger.info(f"Copied {src} to {dest}")
    else:
        logger.error(f"*** Source file {src} does not exist. Skipping copy.")


def go(args):
    
    logger.info("Starting deployment process")
    

    # Define base paths
    # --------------------------------------    
    # Get project root
    project_root = get_project_root(logger)
    logger.info(f"Project root directory: {project_root}")
    
    # Load configuration
    config_filepath = os.path.join(project_root, args.config_file)
    if not os.path.exists(config_filepath):
        logger.error(f"\n***Configuration file {config_filepath} does not exist. Exiting.\n")
        return
    logger.info(f"Loading configuration from: {config_filepath}")    
    config = load_config(config_filepath, logger)
    logger.info(f"Configuration loaded: {config}")
    

    # Define paths of source and destination
    # --------------------------------------    
    # Get the model file path
    latest_model_file = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        'trainedmodel.pkl'
    )
    logger.info(f"Model file path: {latest_model_file}")
    
    
    # Get the score file path
    latest_score_file = os.path.join(
        project_root,
        '02_training',
        config['output_model_path'],
        'latestscore.txt'
    )
    logger.info(f"Latest score file: {latest_score_file}")

    # Get the ingest files record
    ingest_files_record = os.path.join(
        project_root,
        '01_data',
        config['output_folder_path'],
        'ingested_files.txt'
    )
    logger.info(f"Ingest file record: {ingest_files_record}")

    # Get the deployment path
    prod_deployment_path = os.path.join(
        project_root,
        '04_deployment',
        config['prod_deployment_path']        
    )
    logger.info(f"Deployment path: {prod_deployment_path}")


    

    # Copy files to production deployment path
    # --------------------------------------    
    # Copy latest model file   
    logger.info(f"Copying latest model file: {latest_model_file}")
    copy_file(latest_model_file, prod_deployment_path) 
    
    # Copy latest score file    
    logger.info(f"Copying latest score file: {latest_score_file}")
    copy_file(latest_score_file, prod_deployment_path) 
    
    # Copy ingest files record    
    logger.info(f"Copying latest ingest record: {ingest_files_record}")
    copy_file(ingest_files_record, prod_deployment_path) 


    # Log completion
    # --------------------------------------
    logger.info("-----Deployment process completed.-----")

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



