
"""
This script performs model explainability using SHAP (Shapley Additive exPlanations).
It loads the best-performing models from the training phase and generates SHAP plots
(Summary, Bar, and Force plots) to interpret global and local feature importance.
These insights help understand what drives fraudulent predictions.
"""

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import sys
# Import specific model types used for isinstance checks for explainer selection
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# Define paths
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/'

# Ensure reports directory exists for saving plots
os.makedirs(REPORTS_PATH, exist_ok=True)

print("--- Starting Model Explainability (SHAP) ---")

# --- 1. Load Training Results ---
print("Loading training results and best models...")
try:
    # Load the results saved from model_training.py, which include X_test, y_test, and trained pipelines.
    ecommerce_results = joblib.load(os.path.join(MODELS_PATH, 'ecommerce_training_results.pkl'))
    bank_results = joblib.load(os.path.join(MODELS_PATH, 'bank_training_results.pkl'))
    print("Training results loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading training results: {e}. Please ensure 'model_training.py' has been run successfully and output files exist.")
    sys.exit(1)

def generate_shap_plots(model_pipeline, X_test, y_test, dataset_name, model_name):
    """
    Generates and saves SHAP (Shapley Additive exPlanations) plots for a given trained model.
    Plots include: Summary (dot), Bar (feature importance), and Force (individual prediction).

    Args:
        model_pipeline (imblearn.pipeline.Pipeline): The trained model pipeline,
                                                     containing a preprocessor and a classifier.
        X_test (pd.DataFrame): The original (untransformed) test features DataFrame.
                               Used to correctly transform data for the explainer.
        y_test (pd.Series): The true labels for the test set, used to find sample fraudulent instances.
        dataset_name (str): A descriptive name for the dataset (e.g., 'E-commerce Data').
        model_name (str): Name of the model being explained (e.g., 'LightGBM').
    """
    print(f"\nGenerating SHAP plots for {model_name} on {dataset_name}...")

    # Extract the preprocessor and the classifier from the pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']

    # Transform X_test using the preprocessor to prepare data for SHAP explainer.
    # The preprocessor was fit on X_train during model training, so we just transform X_test.
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for the transformed data.
    # ColumnTransformer's `get_feature_names_out()` helps retrieve names for the transformed features.
    numerical_features_orig = X_test.select_dtypes(include=np.number).columns
    categorical_features_orig = X_test.select_dtypes(include=['object', 'bool', 'category']).columns

    # Ensure preprocessor has been fitted (should be from loaded pipeline).
    # `get_feature_names_out` works robustly after fit.
    numerical_feature_names_out = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features_orig)
    categorical_feature_names_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_orig)
    
    # Concatenate all feature names in the order they appear in the transformed output.
    all_transformed_features = np.concatenate([numerical_feature_names_out, categorical_feature_names_out])

    # Convert the transformed test data back to a DataFrame for SHAP, with correct column names.
    # This DataFrame `X_test_transformed_df` is what the SHAP explainer will process.
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_transformed_features, index=X_test.index)

    # Determine the appropriate SHAP Explainer type based on the classifier.
    # TreeExplainer is efficient for tree-based models (like LightGBM).
    # KernelExplainer is model-agnostic but can be computationally intensive; suitable for Logistic Regression.
    if isinstance(classifier, lgb.LGBMClassifier):
        explainer = shap.TreeExplainer(classifier)
        # For binary classification with TreeExplainer, shap_values returns a list of arrays.
        # We are interested in the SHAP values for the positive class (fraud=1).
        shap_values_raw = explainer.shap_values(X_test_transformed_df)
        shap_values_to_plot = shap_values_raw[1] # Index 1 is for the positive class.
        expected_value = explainer.expected_value[1] # Expected value for the positive class.
    elif isinstance(classifier, LogisticRegression):
        print("Using KernelExplainer for Logistic Regression (can be slow for large datasets).")
        # KernelExplainer requires a background dataset to estimate feature contributions.
        # It's ideal to use a representative sample of the transformed training data.
        # Here, a sample from the transformed test set is used for demonstration.
        background_data_sample = shap.utils.sample(X_test_transformed_df, 100) if X_test_transformed_df.shape[0] > 100 else X_test_transformed_df
            
        # KernelExplainer for predict_proba provides SHAP values for each class output.
        explainer = shap.KernelExplainer(classifier.predict_proba, background_data_sample)
        shap_values_raw = explainer.shap_values(X_test_transformed_df)
        shap_values_to_plot = shap_values_raw[1] # Index 1 is for the positive class.
        expected_value = explainer.expected_value[1] # Expected value for the positive class.
    else:
        print(f"Warning: SHAP explainer not explicitly configured for model type: {type(classifier)}. Skipping SHAP plots.")
        return

    # Generate filename-safe strings for saving plots.
    filename_safe_dataset = dataset_name.replace(" ", "_").replace("-", "")
    filename_safe_model = model_name.replace(" ", "_")

    # SHAP Summary Plot (Dot Plot) - Global Feature Importance
    # Shows how features affect the model output, indicating direction and magnitude.
    print("Saving SHAP Summary Dot Plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values_to_plot, X_test_transformed_df, show=False)
    plt.title(f'SHAP Summary Plot - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f'shap_summary_dot_plot_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close()

    # SHAP Bar Plot - Global Feature Importance (Mean Absolute SHAP Value)
    # A simplified view of overall feature importance based on average impact.
    print("Saving SHAP Bar Plot (Feature Importance)...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values_to_plot, X_test_transformed_df, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f'shap_bar_plot_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close()

    # SHAP Force Plot for a sample fraudulent instance - Local Explainability
    # Visualizes how individual feature values contribute to a single prediction,
    # pushing the prediction higher or lower than the base value.
    print("Saving SHAP Force Plot for a sample instance (first fraudulent transaction)...")
    # Find the index of a fraudulent transaction in the original test set
    fraud_indices_in_test = y_test[y_test == 1].index
    if not fraud_indices_in_test.empty:
        # Get the original index to locate the corresponding row in the transformed DataFrame
        original_sample_idx = fraud_indices_in_test[0]
        # Ensure the index exists in the transformed DataFrame (might be dropped if all features were NaN or similar)
        if original_sample_idx in X_test_transformed_df.index:
            transformed_sample_row = X_test_transformed_df.loc[original_sample_idx]
            
            # Calculate SHAP values for this specific single instance
            single_instance_shap_values = explainer.shap_values(transformed_sample_row.to_frame().T)
            
            # Extract positive class SHAP values if explainer returns a list
            if isinstance(single_instance_shap_values, list) and len(single_instance_shap_values) > 1:
                single_instance_shap_values = single_instance_shap_values[1]
            
            # Ensure the SHAP values are 1D array for the force plot (if they came as [[...]])
            if single_instance_shap_values.ndim > 1:
                single_instance_shap_values = single_instance_shap_values[0]

            html_file_path = os.path.join(REPORTS_PATH, f'shap_force_plot_instance_{filename_safe_dataset}_{filename_safe_model}.html')
            
            shap.initjs() # Initialize JavaScript for rendering interactive SHAP plots in HTML
            # Save the force plot as an HTML file.
            shap.save_html(html_file_path, shap.force_plot(expected_value, single_instance_shap_values, transformed_sample_row))
            print(f"SHAP Force Plot for a sample fraudulent instance saved to: {html_file_path}")
        else:
            print(f"Skipping Force Plot: Sample fraudulent instance with original index {original_sample_idx} not found in transformed test set.")
    else:
        print("No fraudulent instances found in test set to generate a Force Plot example for this dataset.")

    print(f"SHAP plots saved for {model_name} on {dataset_name}.")


# --- Process E-commerce Data ---
# Select the best model and its associated test data from the training results.
ecommerce_best_model_name = ecommerce_results['best_model_name']
ecommerce_model_pipeline = ecommerce_results['lr_pipeline'] if ecommerce_best_model_name == 'Logistic Regression' else ecommerce_results['lgbm_pipeline']
ecommerce_X_test = ecommerce_results['X_test']
ecommerce_y_test = ecommerce_results['y_test']

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