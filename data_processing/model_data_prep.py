import numpy as np
from sklearn.preprocessing import OneHotEncoder


def process_data(
    df, label, categorical_features=[], training=True, encoder=None
):
    """ Process the data used in the machine learning pipeline.

    Expects the input DataFrame to contain features and a label column. -->
    Separates the features and label, applies one-hot encoding to categorical features,
    and returns the processed features and label as Numpy arrays.
    This can be used in either training or inference/validation.

    
    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])    
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.    
    """

    # Remove target column from features
    y = df[label]
    X = df.drop([label], axis=1)
    

    # Process the features
    # --------------------
    # Separate categorical and continuous features
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1).values

    # Encode categorical features
    # If training, fit the encoder; otherwise, transform using the existing encoder
    if training is True:        
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")        
        X_categorical = encoder.fit_transform(X_categorical)        
    else:
        X_categorical = encoder.transform(X_categorical)
    

    # Process the label
    # --------------------

    # Convert continuous features to NumPy array
    y = y.values
        
    # Concatenate continuous and categorical features    
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder