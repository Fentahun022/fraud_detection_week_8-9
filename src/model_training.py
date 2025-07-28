

"""
This script handles the core machine learning pipeline for fraud detection,
including data preparation, handling class imbalance using SMOTE,
training and evaluating two models (Logistic Regression and LightGBM),
and saving the trained models and their performance artifacts.
It processes both e-commerce and bank transaction datasets.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn's pipeline for resampling steps
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve, auc
)
from collections import Counter
import joblib # For saving models and preprocessors
import sys

# Import utility functions for modularity
from src.utils import get_feature_types


# Define paths
PROCESSED_DATA_PATH = 'data/processed/'
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/'

# Create directories if they don't exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

print("--- Starting Model Building and Training ---")

# --- 1. Load Engineered Data ---
print("Loading engineered data...")
try:
    # Load the processed and engineered dataframes
    fraud_data_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_engineered.csv'))
    creditcard_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_engineered.csv'))
    print("Engineered data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading engineered file: {e}. Please ensure 'feature_engineering.py' has been run successfully and output files exist.")
    sys.exit(1)


def evaluate_model(y_true, y_pred, y_prob, model_name, dataset_name):
    """
    Evaluates a classification model using F1-score, AUC-PR, and Confusion Matrix.
    Generates and saves plots for the Confusion Matrix and Precision-Recall Curve.

    Args:
        y_true (array-like): True labels of the target variable.
        y_pred (array-like): Predicted labels from the model.
        y_prob (array-like): Predicted probabilities for the positive class (class 1).
        model_name (str): Name of the model being evaluated (e.g., 'Logistic Regression').
        dataset_name (str): Name of the dataset (e.g., 'E-commerce Data').

    Returns:
        dict: A dictionary containing the computed F1-score, AUC-PR, and the confusion matrix.
    """
    print(f"\n--- {model_name} Performance on {dataset_name} ---")

    # F1-Score: A robust metric for imbalanced classification.
    # It is the harmonic mean of precision and recall, balancing false positives and false negatives.
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # AUC-PR (Area Under the Precision-Recall Curve): Highly recommended for imbalanced datasets
    # as it focuses on the performance of the positive (minority) class.
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"AUC-PR: {pr_auc:.4f}")

    # Confusion Matrix: Provides a detailed breakdown of model predictions versus actuals.
    # This matrix is crucial for understanding the types of errors the model makes (FN vs FP),
    # which directly informs business trade-offs in fraud detection.
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix and save to reports/ directory.
    fig, ax = plt.subplots(figsize=(6, 5))
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cmp.plot(ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.tight_layout()
    # Create safe filenames by replacing spaces/hyphens
    filename_safe_dataset = dataset_name.replace(" ", "_").replace("-", "")
    filename_safe_model = model_name.replace(" ", "_")
    plt.savefig(os.path.join(REPORTS_PATH, f'confusion_matrix_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close() # Close plot to free up memory and prevent display in script execution environments

    # Plot Precision-Recall Curve and save to reports/ directory.
    # This curve visually represents the trade-off between precision and recall for different thresholds.
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_curve_auc = auc(recall, precision) # Calculate area under the plotted curve

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {pr_curve_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_PATH, f'pr_curve_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close()

    return {'f1_score': f1, 'auc_pr': pr_auc, 'confusion_matrix': cm}


def process_and_train_dataset(df, target_column_name, dataset_name):
    """
    Orchestrates the data preparation, train-test splitting, feature preprocessing,
    SMOTE application, model training, and evaluation for a single dataset.
    It trains both Logistic Regression and LightGBM models.

    Args:
        df (pd.DataFrame): The input DataFrame for the dataset (e.g., engineered fraud data).
        target_column_name (str): The name of the target variable column (e.g., 'class', 'Class').
        dataset_name (str): A descriptive name for the dataset (e.g., 'E-commerce Data', 'Bank Data').

    Returns:
        dict: A dictionary containing key results for explainability:
              - 'best_model_name' (str): The name of the model performing best based on AUC-PR.
              - 'X_test' (pd.DataFrame): The original (untransformed) test features DataFrame.
              - 'y_test' (pd.Series): The true labels for the test set.
              - 'lr_pipeline' (ImbPipeline): The trained Logistic Regression pipeline.
              - 'lgbm_pipeline' (ImbPipeline): The trained LightGBM pipeline.
    """
    print(f"\n--- Processing and Training Models for {dataset_name} ---")

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Display initial class distribution for awareness of imbalance
    print(f"Class distribution in {dataset_name} before splitting: {Counter(y)}")

    # Drop identifier columns specific to E-commerce data
    # These columns ('user_id', 'device_id', 'ip_address') are not direct features
    # but were used for feature engineering or are unique identifiers.
    if dataset_name == 'E-commerce Data':
        features_to_drop_identifiers = ['user_id', 'device_id', 'ip_address']
        X = X.drop(columns=[col for col in features_to_drop_identifiers if col in X.columns])
    
    # Identify numerical and categorical features for the ColumnTransformer.
    # Using the utility function for modularity and readability.
    numerical_features, categorical_features = get_feature_types(X)

    print(f"Numerical features for {dataset_name} identified for preprocessor: {list(numerical_features)}")
    print(f"Categorical features for {dataset_name} identified for preprocessor: {list(categorical_features)}")

    # Create a preprocessing pipeline using ColumnTransformer.
    # StandardScaler: Scales numerical features to a standard normal distribution (mean=0, std=1).
    #                 Important for distance-based models like Logistic Regression.
    # OneHotEncoder: Converts categorical features into a binary (one-hot) representation.
    #                `handle_unknown='ignore'` prevents errors if new categories appear in test data.
    # `remainder='drop'`: Crucially, any columns not explicitly specified in `transformers` will be dropped.
    #                     This ensures no unexpected dtypes (e.g., lingering objects/strings) reach the models,
    #                     preventing errors like 'could not convert string to float'.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Ensures only specified features are passed through
    )

    # Train-Test Split: Divide data into training and testing sets.
    # `test_size=0.2`: 20% of data for testing, 80% for training.
    # `random_state=42`: For reproducibility of the split.
    # `stratify=y`: Ensures that the proportion of target classes (fraud vs. legitimate)
    #               is the same in both training and testing sets, vital for imbalanced data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Class distribution in {dataset_name} training set: {Counter(y_train)}")
    print(f"Class distribution in {dataset_name} test set: {Counter(y_test)}")

    # --- Model Selection and Training ---
    model_performance_results = {}

    # 1. Logistic Regression (Baseline Model)
    print(f"\nTraining Logistic Regression for {dataset_name}...")
    # ImbPipeline: Integrates preprocessing, sampling, and classifier into one cohesive pipeline.
    # 'preprocessor': Applies scaling and encoding.
    # 'sampler' (SMOTE): Generates synthetic samples for the minority class in the training data.
    #                    `random_state` for reproducibility of synthetic samples.
    # 'classifier': Logistic Regression model.
    #               `solver='liblinear'`: Good for small datasets and handles L1/L2 regularization.
    #               `class_weight='balanced'`: Automatically adjusts weights inversely proportional
    #                                           to class frequencies, emphasizing the minority class.
    #               `max_iter=1000`: Increase iterations for convergence if needed.
    lr_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train) # Pipeline handles preprocessing and sampling internally

    # Predict on the test set (unseen data)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1] # Get probabilities for the positive class

    # Evaluate the Logistic Regression model
    model_performance_results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression', dataset_name)
    # Save the trained pipeline for later use (e.g., explainability, deployment)
    joblib.dump(lr_pipeline, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_").replace("-", "")}_logistic_regression.pkl'))


    # 2. LightGBM (Powerful Ensemble Model)
    print(f"\nTraining LightGBM for {dataset_name}...")
    # ImbPipeline for LightGBM, similar structure to Logistic Regression.
    # 'classifier': LightGBM classifier.
    #               `objective='binary'`: For binary classification task.
    #               `is_unbalance=True`: LightGBM's internal mechanism to handle imbalanced data
    #                                     by focusing more on positive samples or large gradient samples.
    #                                     This works synergistically with SMOTE.
    lgbm_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42, objective='binary', is_unbalance=True))
    ])
    lgbm_pipeline.fit(X_train, y_train)

    # Predict and evaluate LightGBM model
    y_pred_lgbm = lgbm_pipeline.predict(X_test)
    y_prob_lgbm = lgbm_pipeline.predict_proba(X_test)[:, 1]

    model_performance_results['LightGBM'] = evaluate_model(y_test, y_pred_lgbm, y_prob_lgbm, 'LightGBM', dataset_name)
    # Save the trained LightGBM pipeline
    joblib.dump(lgbm_pipeline, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_").replace("-", "")}_lightgbm.pkl'))


    # Justify model selection based on primary metrics (AUC-PR and F1-score for minority class)
    print(f"\n--- Model Selection Justification for {dataset_name} ---")
    best_model_name = None
    if model_performance_results['LightGBM']['auc_pr'] > model_performance_results['Logistic Regression']['auc_pr']:
        best_model_name = 'LightGBM'
    else:
        best_model_name = 'Logistic Regression'

    print(f"Based on AUC-PR, the best performing model for {dataset_name} is: {best_model_name}")
    print("Justification: AUC-PR (Area Under the Precision-Recall Curve) is a preferred metric for "
          "imbalanced datasets over ROC-AUC. It focuses on the positive class and is less prone to "
          "optimistic interpretations when the negative class is dominant. A higher AUC-PR indicates "
          "better performance in distinguishing the minority (fraudulent) class. F1-score also provides "
          "a balance between Precision and Recall, both critical for fraud detection where false positives "
          "(customer inconvenience) and false negatives (financial loss) are costly.")
    
    return {
        'best_model_name': best_model_name,
        'X_test': X_test, # Return original X_test (untransformed) for SHAP explainability.
                          # SHAP explainer will use the pipeline's preprocessor.
        'y_test': y_test,
        'lr_pipeline': lr_pipeline,
        'lgbm_pipeline': lgbm_pipeline
    }


# --- Run for both datasets ---
print("\n--- Training Models for E-commerce Data ---")
ecommerce_results = process_and_train_dataset(fraud_data_df.copy(), 'class', 'E-commerce Data')

print("\n--- Training Models for Bank Transaction Data ---")
# For creditcard_df, target column is 'Class' (capital C) as per dataset.
bank_results = process_and_train_dataset(creditcard_df.copy(), 'Class', 'Bank Data')

print("\n--- Model Training Complete ---")

# Save results containing best model, X_test, and y_test for the explainability script.
joblib.dump(ecommerce_results, os.path.join(MODELS_PATH, 'ecommerce_training_results.pkl'))
joblib.dump(bank_results, os.path.join(MODELS_PATH, 'bank_training_results.pkl'))
print(f"Training results (including best models, X_test, y_test) saved to '{MODELS_PATH}' for explainability.")