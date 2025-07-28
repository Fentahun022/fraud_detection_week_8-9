

"""
Utility functions for the Fraud Detection project.
This module contains reusable helper functions used across different stages
of the data science pipeline (e.g., preprocessing, modeling).
"""
import pandas as pd
import numpy as np

def get_feature_types(X_df):
    """
    Identifies and separates numerical and categorical features from a DataFrame.

    Args:
        X_df (pd.DataFrame): The DataFrame containing features.

    Returns:
        tuple: A tuple containing two lists:
               - numerical_features (list): Column names identified as numerical.
               - categorical_features (list): Column names identified as categorical.
    """
    numerical_features = X_df.select_dtypes(include=np.number).columns.tolist()
    # Include 'object' for strings, 'bool' for booleans, and 'category' for pandas Category dtype
    categorical_features = X_df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    
    return numerical_features, categorical_features