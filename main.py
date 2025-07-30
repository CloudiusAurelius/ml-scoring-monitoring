import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

# Set MLflow tracking URI, experiment, and autologging
mlflow.set_experiment("MLScoringMonitoringExperiment")
mlflow.autolog()

# Define the steps to be executed in the pipeline
_steps = [
    "data_ingestion",
    #"basic_cleaning",
    #"data_check",
    #"data_split",
    #"train_random_forest",
    # Note: use this step only after the model has been trained
    # "test_regression_model"    
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    #os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    #os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        
        if "data_ingestion" in active_steps:
            # Ingest data from the input folder and save it to the output folder
            # This will also log the input and output paths to W&B            
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "01_data"),
                entry_point="main",
                #version='main',
                env_manager="conda",
                parameters={
                    "config_file": config["data_ingestion"]["config_file"],
                    "output_filename": config["data_ingestion"]["output_filename"]
                }
            )
        """
        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
        """
            
        """
        if "basic_cleaning" in active_steps:
            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with outliers and null values removed",
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
                },
            )
        """


        


if __name__ == "__main__":
    go()
