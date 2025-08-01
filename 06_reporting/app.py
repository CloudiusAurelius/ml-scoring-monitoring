from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
#import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import json
import os

import requests
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


# Define paths of source and destination
# --------------------------------------    
# Get project root
project_root = get_project_root(logger)
logger.info(f"Project root directory: {project_root}")

# Load configuration
config_filepath = os.path.join(project_root, 'config.json')
config = load_config(config_filepath, logger)
logger.info(f"Configuration loaded: {config}")

# Set the dataset path
dataset_csv_path = os.path.join(
    project_root,'01_data'
)       

# Set filepath of ingested data and load data
ingested_data_file_path = os.path.join(
        dataset_csv_path,
        config["output_folder_path"],
        'finaldata.csv'
)
df_ingested = load_dataset(ingested_data_file_path, logger)

# Set filepath of test data and load data
test_data_file_path = os.path.join(
        dataset_csv_path,
        config["test_data_path"],
        'testdata.csv'
)
df_test = load_dataset(test_data_file_path, logger)

# Set the model path
prod_deployment_path = os.path.join(
        project_root,
        '04_deployment',
        config['prod_deployment_path']        
)
model_filepath = os.path.join(prod_deployment_path, 'trained_model.pkl')
logger.info(f"Model file path: {model_filepath}")





#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """
    Predict the target variable using the deployed model and data from
    a location provided by the user.
    """
    # Construct filename from request
    filename = request.args.get('filepath', default='/ingesteddata/finaldata.csv', type=str)
    data_file_path = os.path.join(dataset_csv_path, filename)
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
def stats():        
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
def stats():
    """
    Check dependencies, timing and percent NA values in the ingested data.
    """        

    # Check depdenencies
    dependencies_diagnosis=check_outdated_packages(ingested_data_file_path)

    # Check timing
    run_time=execution_time()

    # Check missing values percent
    missing_values_result = missing_values_percent(ingested_data_file_path)

    return jsonify({
        "execution_time": run_time,
        "dependencies_diagnosis": dependencies_diagnosis,
        "missing_values_percent": missing_values_result
    }) #add return value for all diagnostics
   


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)








######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    return #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    return #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    return #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
