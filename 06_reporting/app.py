from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
#import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
#import json
import os

#import requests
import logging

from utils.common_utilities\
    import get_project_root,\
           load_config,\
           load_dataset

from diagnostics.diagnostics\
    import model_predictions,\
           dataframe_summary,\
           execution_time,\
           check_outdated_packages,\
           missing_values_percent,\
           compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


# Define variables
# --------------------------------------
config_file = 'config.json'
ingested_files = 'ingestedfiles.txt'
ingested_data = 'finaldata.csv'
test_data = 'testdata.csv'
model_file = 'trainedmodel.pkl'


# Define paths of source and destination
# --------------------------------------    
# Get project root
project_root = get_project_root(logger)
logger.info(f"Project root directory: {project_root}")

# Load configuration
config_filepath = os.path.join(project_root, config_file)
config = load_config(config_filepath, logger)
logger.info(f"Configuration loaded: {config}")

# Set the dataset path
dataset_csv_path = os.path.join(
    project_root,'01_data'
)
logger.info(f"Dataset CSV path: {dataset_csv_path}")       

# Set filepath of ingested data and load data
logger.info("Loading ingested data")
ingested_data_file_path = os.path.join(
        dataset_csv_path,
        config["output_folder_path"],
        ingested_data
)
if not os.path.exists(ingested_data_file_path):
    logger.error(f"Ingested data file {ingested_data_file_path} does not exist. Exiting.")
    raise FileNotFoundError(f"Ingested data file {ingested_data_file_path} does not exist.")
    exit(1)
df_ingested = load_dataset(ingested_data_file_path, logger)
logger.info(f"Ingested data loaded from {ingested_data_file_path} with shape {df_ingested.shape}")

# Set filepath of test data and load data
logger.info("Loading test data")
test_data_file_path = os.path.join(
        dataset_csv_path,
        config["test_data_path"],
        test_data
)
if not os.path.exists(test_data_file_path):
    logger.error(f"Test data file {test_data_file_path} does not exist. Exiting.")
    raise FileNotFoundError(f"Test data file {test_data_file_path} does not exist.")
    exit(1)
df_test = load_dataset(test_data_file_path, logger)
logger.info(f"Test data loaded from {test_data_file_path} with shape {df_test.shape}")


# Set the model path
logger.info("Setting model file path")
prod_deployment_path = os.path.join(
        project_root,
        '04_deployment',
        config['prod_deployment_path']        
)
model_filepath = os.path.join(prod_deployment_path, model_file)
logger.info(f"Model file path: {model_filepath}")
if not os.path.exists(model_filepath):
    logger.error(f"Model file {model_filepath} does not exist. Exiting.")
    raise FileNotFoundError(f"Model file {model_filepath} does not exist.")
    exit(1)




#######################Welcome Endpoint
@app.route("/", methods=['GET','OPTIONS'])
def welcome():
    """
    Welcome message for the API.
    """
    return jsonify({"message": "Welcome to the ML Model API!"})


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """
    Predict the target variable using the deployed model and data from
    a location provided by the user.
    """
    # Construct filename from request
    #default_route = f"/{config['output_folder_path']}/{ingested_data}"
    #filename = request.args.get('filepath')
    filename = request.get_json().get('filepath')
    logger.info(f"Received request to predict with file: {filename}")
    #data_file_path = os.path.join(dataset_csv_path, filename)
    data_file_path = dataset_csv_path+str(filename)
    logger.info(f"Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Load the dataset
    logging.info(f"Loading dataset from: {data_file_path}")
    df = pd.read_csv(data_file_path)
    logger.info(f"Dataset loaded from {data_file_path} with shape {df.shape}")

    # Make predictions
    logger.info("Making predictions on the input data and extracting the true labels")
    y_pred, y_true = model_predictions(df, model_filepath)
    
    return jsonify({"predictions": y_pred, "true_labels": y_true})



#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    """
    Check the f1 score of the deployed model on the test dataset.
    """
    
    # Make prediction with deployed model on test data
    logger.info("Scoring the deployed model on the test data")
    y_pred, y_true = model_predictions(df_test, model_filepath)
    
    # Convert to numpy arrays for metric calculation
    logger.info("Converting predictions and true labels to numpy arrays")
    y_true_np= np.array(y_true)
    y_pred_np= np.array(y_pred)

    # Compute model metrics
    # ---------------------------------------   
    logger.info("Computing model metrics")
    _,\
    _,\
    fbeta,\
    _ = compute_model_metrics(y_true_np, y_pred_np)

    return jsonify({"f1_score": fbeta})#add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    """
    Check means, medians, and modes for each column in the ingested data.
    """   

    # Compute summary statistics
    logger.info("Calculating summary statistics for the ingested data")
    summary_stats = dataframe_summary(df_ingested)
    
    logger.info(f"Summary statistics calculated: {summary_stats}")
    return jsonify(summary_stats)  
    

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    Check dependencies, timing and percent NA values in the ingested data.
    """        

    # Check depdenencies
    logger.info("Checking outdated packages")
    dependencies_diagnosis=check_outdated_packages()

    # Check timing
    logger.info("Checking execution time of the ingested data")
    run_time=execution_time()

    # Check missing values percent
    logger.info("Checking missing values percent in the ingested data")
    missing_values_result = missing_values_percent(df_ingested)

    return jsonify({
        "execution_time": run_time,
        "dependencies_diagnosis": dependencies_diagnosis\
                                  .to_dict(orient='records'),
        "missing_values_percent": missing_values_result
    }) #add return value for all diagnostics
   


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)





