# src/model_training.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib # For saving models and preprocessors

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
    print(f"Error loading engineered file: {e}. Please ensure 'feature_engineering.py' has been run successfully.")
    exit()


# --- Define Evaluation Function ---
def evaluate_model(y_true, y_pred, y_prob, model_name, dataset_name):
    """Evaluates a classification model using F1-score, AUC-PR, and Confusion Matrix."""
    print(f"\n--- {model_name} Performance on {dataset_name} ---")

    # F1-Score
    f1 = f1_score(y_true, y_pred) # [2, 3, 6, 7, 15]
    print(f"F1-Score: {f1:.4f}")

    # AUC-PR (Area Under the Precision-Recall Curve)
    # average_precision_score directly computes the PR-AUC for binary classification [37, 38, 49]
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"AUC-PR: {pr_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) # [1, 11, 14, 20, 21]
    cmp.plot(ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f'confusion_matrix_{dataset_name}_{model_name.replace(" ", "_")}.png'))
    plt.close() # Close plot to prevent display issues in script execution

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_curve_auc = auc(recall, precision) # [38, 46, 48, 49]

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {pr_curve_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_PATH, f'pr_curve_{dataset_name}_{model_name.replace(" ", "_")}.png'))
    plt.close()

    return {'f1_score': f1, 'auc_pr': pr_auc, 'confusion_matrix': cm}


# --- Function to process and train a single dataset ---
def process_and_train_dataset(df, target_column_name, dataset_name):
    print(f"\n--- Processing and Training Models for {dataset_name} ---")

    # Separate features and target
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Analyze Class Imbalance
    print(f"Class distribution in {dataset_name} before splitting: {Counter(y)}")
    # The prompt explicitly mentions class imbalance as a critical challenge.
    # We will use stratified split and then apply sampling on training data. [10, 12, 23, 44, 45]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Remove user_id, device_id, ip_address (original string) as they are identifiers or already converted
    # 'ip_address_int' is numerical and should be kept.
    # For 'Fraud_Data', 'user_id', 'device_id', 'ip_address' are identifiers or raw.
    # For 'creditcard', no such columns need removal.
    if 'user_id' in numerical_features:
        numerical_features = numerical_features.drop('user_id')
    if 'device_id' in categorical_features: # device_id is object
        categorical_features = categorical_features.drop('device_id')
    if 'ip_address' in categorical_features: # ip_address is object
        categorical_features = categorical_features.drop('ip_address')
    
    # Exclude time-related datetime objects if they still exist after feature engineering
    if 'signup_time' in X.columns:
        X = X.drop(columns=['signup_time'])
        numerical_features = [col for col in numerical_features if col != 'signup_time']
    if 'purchase_time' in X.columns:
        X = X.drop(columns=['purchase_time'])
        numerical_features = [col for col in numerical_features if col != 'purchase_time']
    
    # Re-evaluate numerical and categorical features after dropping identifiers
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    print(f"Numerical features for {dataset_name}: {list(numerical_features)}")
    print(f"Categorical features for {dataset_name}: {list(categorical_features)}")

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler() # [4, 17, 19, 24, 28]
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # [5, 8, 16, 29, 32]

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though none expected here
    )

    # Train-Test Split (Stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # [10, 12]
    print(f"Class distribution in {dataset_name} training set: {Counter(y_train)}")
    print(f"Class distribution in {dataset_name} test set: {Counter(y_test)}")

    # Handle Class Imbalance on Training Data Only
    # Research and apply appropriate sampling techniques (e.g., SMOTE, Random Undersampling)
    # Justify your choice: SMOTE (oversampling minority) and/or RandomUnderSampler (undersampling majority)
    # SMOTE is often preferred as it generates synthetic samples, avoiding information loss from undersampling. [13, 25, 31, 33, 43]
    # For highly imbalanced datasets, a combination might be needed, or using `scale_pos_weight` in tree models.
    
    # For this project, we'll demonstrate SMOTE for oversampling.
    print(f"\nApplying SMOTE to {dataset_name} training data...")
    smote = SMOTE(random_state=42) # [25, 31, 33, 43]
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Class distribution in {dataset_name} training set after SMOTE: {Counter(y_train_resampled)}")

    # Alternatively, for very large datasets, RandomUnderSampler can be faster but loses info. [13, 26, 27, 36]
    # rus = RandomUnderSampler(random_state=42)
    # X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)


    # --- Model Selection and Training ---
    results = {}

    # 1. Logistic Regression (Baseline)
    print(f"\nTraining Logistic Regression for {dataset_name}...")
    # Using class_weight='balanced' is crucial for imbalanced data in Logistic Regression [9, 18, 22, 30, 42]
    lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))]) # liblinear is good for small datasets and L1/L2 penalty

    lr_model.fit(X_train_resampled, y_train_resampled)

    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1] # Probability of the positive class

    results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression', dataset_name)
    joblib.dump(lr_model, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_")}_logistic_regression.pkl'))


    # 2. LightGBM (Powerful Ensemble Model)
    print(f"\nTraining LightGBM for {dataset_name}...")
    # LightGBM can handle imbalance via `is_unbalance=True` or `scale_pos_weight` [34, 35, 41, 47, 50]
    # `scale_pos_weight` is often preferred for more control (num_negative / num_positive).
    # We will use scale_pos_weight based on the original imbalanced training data for more direct control,
    # and also because SMOTE might sometimes artificially inflate feature relationships.
    # Alternatively, you can use SMOTE and then not set scale_pos_weight. Let's try it with SMOTE.
    # If not using SMOTE, calculate scale_pos_weight:
    # scale_pos_weight_val = (Counter(y_train)[0] / Counter(y_train)[1]) if Counter(y_train)[1] > 0 else 1.0

    lgbm_model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', lgb.LGBMClassifier(random_state=42, objective='binary',
                                                                    # class_weight='balanced' or scale_pos_weight
                                                                    # We used SMOTE, so we generally don't set this
                                                                    # for LightGBM as SMOTE handles the imbalance
                                                                    # in the training data itself.
                                                                    # If NOT using SMOTE: scale_pos_weight=scale_pos_weight_val
                                                                    is_unbalance=True # Alternative for LightGBM with imbalanced data
                                                                   ))])

    lgbm_model.fit(X_train_resampled, y_train_resampled)

    y_pred_lgbm = lgbm_model.predict(X_test)
    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

    results['LightGBM'] = evaluate_model(y_test, y_pred_lgbm, y_prob_lgbm, 'LightGBM', dataset_name)
    joblib.dump(lgbm_model, os.path.join(MODELS_PATH, f'{dataset_name.lower().replace(" ", "_")}_lightgbm.pkl'))


    # Justify model selection (based on AUC-PR and F1-score for minority class)
    print(f"\n--- Model Selection Justification for {dataset_name} ---")
    best_model_name = None
    if results['LightGBM']['auc_pr'] > results['Logistic Regression']['auc_pr']:
        best_model_name = 'LightGBM'
    else:
        best_model_name = 'Logistic Regression'

    print(f"Based on AUC-PR, the best performing model for {dataset_name} is: {best_model_name}")
    print("Justification: AUC-PR is a suitable metric for imbalanced datasets as it focuses on the positive class, "
          "providing a better understanding of the model's ability to identify fraud cases. "
          "F1-score also balances precision and recall, which is important when both false positives "
          "and false negatives have significant costs.")
    
    return lr_model, lgbm_model, best_model_name, X_test, y_test


# --- Run for both datasets ---
print("\n--- Training Models for E-commerce Data ---")
lr_ecommerce, lgbm_ecommerce, best_ecommerce_model, X_test_ecommerce, y_test_ecommerce = \
    process_and_train_dataset(fraud_data_df.copy(), 'class', 'E-commerce Data')

print("\n--- Training Models for Bank Transaction Data ---")
# For creditcard_df, target column is 'Class' (capital C)
lr_bank, lgbm_bank, best_bank_model, X_test_bank, y_test_bank = \
    process_and_train_dataset(creditcard_df.copy(), 'Class', 'Bank Data')

print("\n--- Model Training Complete ---")