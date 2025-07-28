# src/model_explainability.py

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import sys
# Import specific model types used for isinstance checks
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# Define paths
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/'

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_PATH, exist_ok=True)

print("--- Starting Model Explainability (SHAP) ---")

# --- 1. Load Training Results ---
print("Loading training results and best models...")
try:
    ecommerce_results = joblib.load(os.path.join(MODELS_PATH, 'ecommerce_training_results.pkl'))
    bank_results = joblib.load(os.path.join(MODELS_PATH, 'bank_training_results.pkl'))
    print("Training results loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading training results: {e}. Please ensure 'model_training.py' has been run successfully and output files exist.")
    sys.exit(1)

# Function to generate and interpret SHAP plots
def generate_shap_plots(model_pipeline, X_test, y_test, dataset_name, model_name):
    """
    Generates and saves SHAP Summary (dot and bar) and Force plots for a given model.

    Args:
        model_pipeline (imblearn.pipeline.Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features DataFrame.
        y_test (pd.Series): The true labels for the test set.
        dataset_name (str): Name of the dataset (e.g., 'E-commerce Data').
        model_name (str): Name of the model (e.g., 'LightGBM').
    """
    print(f"\nGenerating SHAP plots for {model_name} on {dataset_name}...")

    # Extract the preprocessor and the classifier from the pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']

    # Get feature names after one-hot encoding for the transformed data
    # Create a dummy DataFrame to fit_transform and get feature names
    # A robust way to get feature names from ColumnTransformer
    numerical_features_orig = X_test.select_dtypes(include=np.number).columns
    categorical_features_orig = X_test.select_dtypes(include=['object', 'bool', 'category']).columns

    # Temporarily fit the preprocessor on X_test to get all feature names
    # This assumes X_test is representative enough to get all possible categories for OHE.
    # For robust production, preprocessor should be fit on X_train.
    # However, since it's already part of the loaded pipeline, we can just transform X_test.
    preprocessor.fit(X_test) # Fit ensures get_feature_names_out works correctly based on categories seen.
                             # This is OK here because we are only getting feature names for plotting,
                             # not for re-fitting the scaler/encoder which was fit on X_train.

    numerical_feature_names_out = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features_orig)
    categorical_feature_names_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_orig)
    
    all_transformed_features = np.concatenate([numerical_feature_names_out, categorical_feature_names_out])

    # Transform X_test using the preprocessor to get SHAP values
    X_test_transformed = preprocessor.transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_transformed_features, index=X_test.index)

    # Determine Explainer type based on classifier
    if isinstance(classifier, lgb.LGBMClassifier):
        explainer = shap.TreeExplainer(classifier)
        # TreeExplainer for binary classification returns a list [shap_values_class_0, shap_values_class_1]
        shap_values_raw = explainer.shap_values(X_test_transformed_df)
        shap_values_to_plot = shap_values_raw[1] # Focus on the positive class (fraud=1)
        expected_value = explainer.expected_value[1] # Expected value for positive class
    elif isinstance(classifier, LogisticRegression):
        print("Using KernelExplainer for Logistic Regression (can be slow for large datasets, using a sample for background data)...")
        # KernelExplainer requires a background dataset. Use a sample from the transformed test set.
        # Ideally, this should be a representative sample of the *transformed training data*.
        if X_test_transformed_df.shape[0] > 500:
            background_data = shap.utils.sample(X_test_transformed_df, 100)
        else:
            background_data = X_test_transformed_df
            
        explainer = shap.KernelExplainer(classifier.predict_proba, background_data)
        # KernelExplainer for predict_proba returns a list of shap_values for each class
        shap_values_raw = explainer.shap_values(X_test_transformed_df)
        shap_values_to_plot = shap_values_raw[1] # Focus on the positive class (fraud=1)
        expected_value = explainer.expected_value[1] # Expected value for positive class
    else:
        print(f"Warning: SHAP explainer not explicitly configured for model type: {type(classifier)}. Skipping SHAP plots.")
        return

    # Generate filename-safe strings for saving plots
    filename_safe_dataset = dataset_name.replace(" ", "_").replace("-", "")
    filename_safe_model = model_name.replace(" ", "_")

    # SHAP Summary Plot (Global Importance) - Dot plot
    print("Saving SHAP Summary Dot Plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values_to_plot, X_test_transformed_df, show=False)
    plt.title(f'SHAP Summary Plot - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f'shap_summary_dot_plot_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close()

    # SHAP Bar Plot (Global Importance - Mean Absolute SHAP Value)
    print("Saving SHAP Bar Plot (Feature Importance)...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values_to_plot, X_test_transformed_df, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f'shap_bar_plot_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close()

    # SHAP Force Plot for a sample instance (e.g., first fraudulent transaction)
    print("Saving SHAP Force Plot for a sample instance (first fraudulent transaction)...")
    fraud_indices_in_test = y_test[y_test == 1].index
    if not fraud_indices_in_test.empty:
        # Get the original index from X_test, then locate it in the transformed DataFrame
        original_sample_idx = fraud_indices_in_test[0]
        # Check if the index exists in X_test_transformed_df after any potential filtering/drops
        if original_sample_idx in X_test_transformed_df.index:
            transformed_sample_row = X_test_transformed_df.loc[original_sample_idx]
            
            # Calculate SHAP values for this specific instance (needs to be 2D array if classifier expects it)
            # explainer.shap_values() can sometimes return 1D or 2D based on internal state
            single_instance_shap_values = explainer.shap_values(transformed_sample_row.to_frame().T)
            
            # Extract positive class SHAP values if necessary
            if isinstance(single_instance_shap_values, list) and len(single_instance_shap_values) > 1:
                single_instance_shap_values = single_instance_shap_values[1]
            
            # Ensure it's 1D for force plot
            if single_instance_shap_values.ndim > 1:
                single_instance_shap_values = single_instance_shap_values[0]

            html_file_path = os.path.join(REPORTS_PATH, f'shap_force_plot_instance_{filename_safe_dataset}_{filename_safe_model}.html')
            
            shap.initjs() # Initialize JavaScript for interactive plots in HTML
            # The expected_value must be a scalar for force_plot
            shap.save_html(html_file_path, shap.force_plot(expected_value, single_instance_shap_values, transformed_sample_row))
            print(f"SHAP Force Plot for a sample fraudulent instance saved to: {html_file_path}")
        else:
            print(f"Skipping Force Plot: Sample fraudulent instance with original index {original_sample_idx} not found in transformed test set.")
    else:
        print("No fraudulent instances found in test set to generate a Force Plot example for this dataset.")

    print(f"SHAP plots saved for {model_name} on {dataset_name}.")


# --- Process E-commerce Data ---
ecommerce_best_model_name = ecommerce_results['best_model_name']
ecommerce_model_pipeline = ecommerce_results['lr_pipeline'] if ecommerce_best_model_name == 'Logistic Regression' else ecommerce_results['lgbm_pipeline']
ecommerce_X_test = ecommerce_results['X_test']
ecommerce_y_test = ecommerce_results['y_test'] # Need this for finding fraudulent instances

print(f"\n--- Explaining Best Model for E-commerce Data: {ecommerce_best_model_name} ---")
generate_shap_plots(ecommerce_model_pipeline, ecommerce_X_test, ecommerce_y_test, 'E-commerce Data', ecommerce_best_model_name)

# --- Process Bank Data ---
bank_best_model_name = bank_results['best_model_name']
bank_model_pipeline = bank_results['lr_pipeline'] if bank_best_model_name == 'Logistic Regression' else bank_results['lgbm_pipeline']
bank_X_test = bank_results['X_test']
bank_y_test = bank_results['y_test']

print(f"\n--- Explaining Best Model for Bank Data: {bank_best_model_name} ---")
generate_shap_plots(bank_model_pipeline, bank_X_test, bank_y_test, 'Bank Data', bank_best_model_name)

print("\n--- Model Explainability Complete ---")