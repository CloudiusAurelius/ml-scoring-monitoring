"""
# utils/common_utilities.py

This module contains common utility functions used across different scripts.
"""

import logging
import json
import os

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