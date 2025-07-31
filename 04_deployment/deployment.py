"""
# 04_deployment/deployment.py

This script is responsible for deploying the trained model by copying:
- the record of the ingest files
- the latest model file
- the latest score file
into the production deployment directory.
"""

import os

from utils.common_utilities\
    import get_project_root,\
           load_config
           

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    logger.info("Starting deployment process")

    # Load configuration
    config = load_config(args.config_file)
    logger.info(f"Configuration loaded from: {args.config_file}")
    
    # Get project root
    project_root = get_project_root()
    
    # Define paths
    model_folder_path = os.path.join(project_root, config['output_folder_path'])
    logger.info(f"Model folder path: {model_folder_path}")
    prod_deployment_path = os.path.join(project_root, '04_deployment', config['prod_deployment_path'])
    logger.info(f"Production deployment path: {prod_deployment_path}")
    
    # Copy latest model file
    latest_model_file = os.path.join(model_folder_path, config['output_modelname'])
    if os.path.exists(latest_model_file):
        logger.info(f"Copying latest model file: {latest_model_file} to {prod_deployment_path}")
        os.system(f"cp {latest_model_file} {prod_deployment_path}")
    logger.info(f"Latest model file copied successfully to: {prod_deployment_path}")


    # Copy latest score file
    latest_score_file = os.path.join(model_folder_path, 'latestscore.txt')
    if os.path.exists(latest_score_file):
        logger.info(f"Copying latest score file: {latest_score_file} to {prod_deployment_path}")
        os.system(f"cp {latest_score_file} {prod_deployment_path}")
    logger.info(f"Latest score file copied successfully to: {prod_deployment_path}")
    
    # Copy ingest files record
    ingest_files_record = os.path.join(model_folder_path, 'ingestfiles.txt')
    if os.path.exists(ingest_files_record):
        logger.info(f"Copying ingest files record: {ingest_files_record} to {prod_deployment_path}")
        os.system(f"cp {ingest_files_record} {prod_deployment_path}")
    logger.info(f"Ingest files record copied successfully to: {prod_deployment_path}")


    logger.info("Deployment process completed successfully.")

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



