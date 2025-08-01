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
    "model_training",
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


        if "model_training" in active_steps:
            # Train the model using the ingested data
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "02_training"),
                entry_point="main",
                #version='main',
                env_manager="conda",
                parameters={
                    "config_file": config["model_training"]["config_file"],
                    "output_modelname": config["model_training"]["output_modelname"]
                }
            )

        if "model_scoring" in active_steps:
            # Train the model using the ingested data
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "03_scoring"),
                entry_point="main",
                #version='main',
                env_manager="conda",
                parameters={
                    "config_file": config["model_scoring"]["config_file"],
                    "input_data": config["model_scoring"]["input_data"],
                    "input_modelinfo": config["model_scoring"]["input_modelinfo"],
                    "output_score_filename": config["model_scoring"]["output_score_filename"]
                }
            )           


        if "model_deployment" in active_steps:
            # Train the model using the ingested data
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "04_deployment"),
                entry_point="main",
                #version='main',
                env_manager="conda",
                parameters={
                    "config_file": config["model_deployment"]["config_file"]
                }
            )           


if __name__ == "__main__":
    go()
