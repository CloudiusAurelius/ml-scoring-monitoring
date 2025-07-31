import pickle


model_path = '/Users/bernd/Documents/Courses/Udacity_MLDevOps/ml-scoring-monitoring/02_training/practicemodels/trainedmodel.pkl'
print(f"Checking if model file exists at: {model_path}")



print(f"Loading model from: {model_path}")
with open(model_path, "rb") as f:
    model_info = pickle.load(f)
    print(model_info)
    model_created_at = model_info["created_at"]
    print(model_created_at)

"""
print(f"Loading model from: {model_path}")
with open(model_path, 'rb') as filehandler:
    model_name = pickle.load(filehandler)["name"]
    model_created_at = pickle.load(filehandler)["created_at"]
    model = pickle.load(filehandler)["model"]
    encoder = pickle.load(filehandler)["encoder"]
    label = pickle.load(filehandler)["label_column"]
    categorical_features = pickle.load(filehandler)["categorical_features"]
print(f"Model name: {model_name}")
print(f"Model created at: {model_created_at}")
print(f"Model: {model}")
print(f"Encoder: {encoder}")
print(f"Label column: {label}")
print(f"Categorical features: {categorical_features}")
print(f"Model type: {type(model)}")
print(f"Encoder type: {type(encoder)}")
print(f"Label type: {type(label)}")
print(f"Categorical features type: {type(categorical_features)}")
print(f"Model params: {model.get_params()}")
"""