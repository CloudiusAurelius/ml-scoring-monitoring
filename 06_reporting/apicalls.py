import requests

from utils.common_utilities\
    import get_project_root,\
           load_config

import logging
import os


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
logger.info("Calling API endpoints")
response1 = requests.post(URL+'/prediction', json={'filepath': '/testdata/testdata.csv'}).content
logger.info({"response1": response1})
logger.info("response1 completed")
response2 = requests.get(URL + '/scoring').content
logger.info("response2 completed")
response3 = requests.get(URL +'/summarystats').content
logger.info("response3 completed")
response4 = requests.get(URL + "/diagnostics").content
logger.info("response4 completed")


#combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

#write the responses to your workspace
project_root = get_project_root(logger)
logger.info(f"Project root directory: {project_root}")

config_filepath = os.path.join(project_root, 'config.json')
config = load_config(config_filepath, logger)
logger.info(f"Configuration loaded: {config}")

output_path = os.path.join(
                project_root,
                '02_training',
                config["output_model_path"],                
                'apireturns.txt'
)        
with open(output_path, 'w') as f:
    for key, value in responses.items():
        f.write(f"{key}:\n{value.decode('utf-8')}\n\n")
logger.info(f"Responses written to {output_path}")