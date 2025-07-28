# src/model_training.py

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
    fraud_data_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_engineered.csv'))
    creditcard_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_engineered.csv'))
    print("Engineered data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading engineered file: {e}. Please ensure 'feature_engineering.py' has been run successfully and output files exist.")
    sys.exit(1)


# --- Define Evaluation Function ---
def evaluate_model(y_true, y_pred, y_prob, model_name, dataset_name):
    """
    Evaluates a classification model using F1-score, AUC-PR, and Confusion Matrix.
    Generates and saves plots for the Confusion Matrix and Precision-Recall Curve.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model being evaluated (e.g., 'Logistic Regression').
        dataset_name (str): Name of the dataset (e.g., 'E-commerce Data').

    Returns:
        dict: A dictionary containing F1-score, AUC-PR, and the confusion matrix.
    """
    print(f"\n--- {model_name} Performance on {dataset_name} ---")

    # F1-Score: Harmonic mean of precision and recall, crucial for imbalanced data.
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # AUC-PR (Area Under the Precision-Recall Curve): Excellent for imbalanced datasets
    # as it focuses on the positive (minority) class.
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"AUC-PR: {pr_auc:.4f}")

    # Confusion Matrix: Provides a detailed breakdown of True Positives, False Positives,
    # True Negatives, and False Negatives, directly informing business trade-offs.
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix and save to reports/
    fig, ax = plt.subplots(figsize=(6, 5))
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cmp.plot(ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.tight_layout()
    # Replace spaces and hyphens for clean filename
    filename_safe_dataset = dataset_name.replace(" ", "_").replace("-", "")
    filename_safe_model = model_name.replace(" ", "_")
    plt.savefig(os.path.join(REPORTS_PATH, f'confusion_matrix_{filename_safe_dataset}_{filename_safe_model}.png'))
    plt.close() # Close plot to prevent display issues in script execution

    # Plot Precision-Recall Curve and save to reports/
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_curve_auc = auc(recall, precision)

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


# --- Function to process and train a single dataset ---
def process_and_train_dataset(df, target_column_name, dataset_name):
    """
    Performs data preparation, splits into train/test sets, applies preprocessing and SMOTE,
    trains Logistic Regression and LightGBM models, and evaluates their performance.

    Args:
        df (pd.DataFrame): The input DataFrame for the dataset.
        target_column_name (str): The name of the target variable column.
        dataset_name (str): A descriptive name for the dataset (e.g., 'E-commerce Data').

    Returns:
        dict: A dictionary containing the best model's name, X_test, y_test,
              and both trained pipeline models for further explainability.
    """
    print(f"\n--- Processing and Training Models for {dataset_name} ---")

    # Separate features and target
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Analyze Class Imbalance
    print(f"Class distribution in {dataset_name} before splitting: {Counter(y)}")

    # Identify features to drop (identifiers) specific to E-commerce data
    # These columns should not be used as features directly but can be important for engineering.
    # 'ip_address_int' and 'country' are kept as engineered features derived from 'ip_address'.
    if dataset_name == 'E-commerce Data':
        features_to_drop_identifiers = ['user_id', 'device_id', 'ip_address']
        # Drop these identifiers from the feature set X
        X = X.drop(columns=[col for col in features_to_drop_identifiers if col in X.columns])
    
    # --- DIAGNOSTIC PRINT ADDED HERE ---
    # This print will help us confirm what columns are present AFTER dropping identifiers.
    print(f"\n[DEBUG] {dataset_name} X.columns AFTER dropping identifiers:")
    print(X.columns.tolist())
    print(f"[DEBUG] {dataset_name} X.dtypes AFTER dropping identifiers:")
    print(X.dtypes)
    print(f"[DEBUG] Check for any lingering 'object' or 'string' dtypes that should be encoded:")
    print(X.select_dtypes(include=['object', 'string']).columns.tolist())
    # --- END DIAGNOSTIC PRINT ---

    # Identify numerical and categorical features for preprocessing pipeline
    # Ensure 'category' dtype is also included for categorical features as some might be converted earlier.
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns

    print(f"Numerical features for {dataset_name} identified for preprocessor: {list(numerical_features)}")
    print(f"Categorical features for {dataset_name} identified for preprocessor: {list(categorical_features)}")

    # Create a preprocessor using ColumnTransformer
    # StandardScaler for numerical features (scaling helps Logistic Regression, good for tree-based too)
    # OneHotEncoder for categorical features (converts categories to numerical format)
    # --- CRITICAL FIX: remainder='drop' ---
    # This ensures that any column not explicitly handled by 'num' or 'cat' transformers
    # will be dropped, preventing unexpected string/object columns from reaching SMOTE.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Changed from 'passthrough' to 'drop' for robustness
    )

    # Train-Test Split (Stratified to maintain class distribution in both sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Class distribution in {dataset_name} training set: {Counter(y_train)}")
    print(f"Class distribution in {dataset_name} test set: {Counter(y_test)}")

    # --- Model Selection and Training ---
    model_performance_results = {}

    # 1. Logistic Regression (Baseline Model)
    # Rationale: Simple, interpretable, and provides a good baseline for comparison.
    # Imbalance handling: `class_weight='balanced'` automatically adjusts weights.
    # SMOTE is also in the pipeline to synthetically balance the minority class,
    # further improving its ability to learn from fraud examples.
    print(f"\nTraining Logistic Regression for {dataset_name}...")
    lr_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)), # Apply SMOTE to the training data only
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train) # Pipeline handles preprocessing and sampling internally

    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (fraud)

    model_performance_results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression', dataset_name)
    # Save the trained pipeline model
    joblib.dump(lr_pipeline, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_").replace("-", "")}_logistic_regression.pkl'))


    # 2. LightGBM (Powerful Ensemble Model)
    # Rationale: Gradient Boosting models (like LightGBM/XGBoost) are highly effective
    # for tabular data, robust to outliers, and handle non-linear relationships well.
    # Imbalance handling: `is_unbalance=True` or `scale_pos_weight` (here using `is_unbalance=True`)
    # combined with SMOTE for a strong approach to imbalanced classification.
    print(f"\nTraining LightGBM for {dataset_name}...")
    lgbm_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)), # Apply SMOTE to the training data only
        ('classifier', lgb.LGBMClassifier(random_state=42, objective='binary', is_unbalance=True))
    ])
    lgbm_pipeline.fit(X_train, y_train) # Pipeline handles preprocessing and sampling internally

    y_pred_lgbm = lgbm_pipeline.predict(X_test)
    y_prob_lgbm = lgbm_pipeline.predict_proba(X_test)[:, 1]

    model_performance_results['LightGBM'] = evaluate_model(y_test, y_pred_lgbm, y_prob_lgbm, 'LightGBM', dataset_name)
    # Save the trained pipeline model
    joblib.dump(lgbm_pipeline, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_").replace("-", "")}_lightgbm.pkl'))


    # Justify model selection (based on AUC-PR and F1-score for minority class)
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
# For creditcard_df, target column is 'Class' (capital C)
bank_results = process_and_train_dataset(creditcard_df.copy(), 'Class', 'Bank Data')

print("\n--- Model Training Complete ---")

# Save results for explainability script (including X_test, y_test, and chosen best model)
joblib.dump(ecommerce_results, os.path.join(MODELS_PATH, 'ecommerce_training_results.pkl'))
joblib.dump(bank_results, os.path.join(MODELS_PATH, 'bank_training_results.pkl'))
print(f"Training results (including best models, X_test, y_test) saved to '{MODELS_PATH}' for explainability.")