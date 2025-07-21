# src/feature_engineering.py

import pandas as pd
import numpy as np
import os

# Define paths
PROCESSED_DATA_PATH = 'data/processed/'

print("--- Starting Feature Engineering ---")

# --- 1. Load Processed Data ---
print("Loading processed data...")
try:
    fraud_data_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_processed.csv'))
    creditcard_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_processed.csv'))
    
    # Ensure datetime columns are loaded as datetime objects again
    fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])
    fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])

    print("Processed data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading processed file: {e}. Please ensure 'data_preprocessing.py' has been run successfully.")
    exit()

# --- 2. Feature Engineering for Fraud_Data.csv ---
print("\n--- Feature Engineering for E-commerce Data (Fraud_Data.csv) ---")

# Time-Based Features
print("Creating time-based features...")
fraud_data_df['hour_of_day'] = fraud_data_df['purchase_time'].dt.hour # [6, 9, 10]
fraud_data_df['day_of_week'] = fraud_data_df['purchase_time'].dt.dayofweek # Monday=0, Sunday=6 [6, 7, 8, 9, 10]
fraud_data_df['day_of_year'] = fraud_data_df['purchase_time'].dt.dayofyear
fraud_data_df['month'] = fraud_data_df['purchase_time'].dt.month
fraud_data_df['year'] = fraud_data_df['purchase_time'].dt.year

# time_since_signup: Calculate the duration between signup_time and purchase_time.
# Convert to seconds for numerical feature [1, 2, 3, 4, 5]
fraud_data_df['time_since_signup_seconds'] = (fraud_data_df['purchase_time'] - fraud_data_df['signup_time']).dt.total_seconds()
# Handle cases where purchase_time might be before signup_time (unlikely for legitimate, but possible for fraudulent)
fraud_data_df['time_since_signup_seconds'] = fraud_data_df['time_since_signup_seconds'].apply(lambda x: max(x, 0))


# Transaction Frequency and Velocity (per user and per device)
# Sort by user_id/device_id and then by purchase_time for correct rolling window calculation
fraud_data_df.sort_values(by=['user_id', 'purchase_time'], inplace=True)
print("Calculating transaction frequency and velocity per user...")

# Number of transactions by user in last 24 hours (86400 seconds)
# This requires a rolling window on time [11, 12, 13, 14, 15]
# Use `groupby` and `rolling` with a time offset
# First, set 'purchase_time' as index for rolling window
fraud_data_temp_user = fraud_data_df[['user_id', 'purchase_time', 'purchase_value']].copy()
fraud_data_temp_user.set_index('purchase_time', inplace=True)
fraud_data_temp_user = fraud_data_temp_user.sort_index() # Ensure index is sorted for rolling window

# Group by user_id and apply rolling window
user_transaction_counts = fraud_data_temp_user.groupby('user_id').rolling('24H')['purchase_value'].count().reset_index()
user_transaction_counts.rename(columns={'purchase_value': 'user_transactions_24hr'}, inplace=True)

user_transaction_values_sum = fraud_data_temp_user.groupby('user_id').rolling('24H')['purchase_value'].sum().reset_index()
user_transaction_values_sum.rename(columns={'purchase_value': 'user_value_24hr_sum'}, inplace=True)

# Merge these back to the original dataframe
fraud_data_df = pd.merge(fraud_data_df, user_transaction_counts, on=['user_id', 'purchase_time'], how='left')
fraud_data_df = pd.merge(fraud_data_df, user_transaction_values_sum, on=['user_id', 'purchase_time'], how='left')

# Repeat for device_id
print("Calculating transaction frequency and velocity per device...")
fraud_data_temp_device = fraud_data_df[['device_id', 'purchase_time', 'purchase_value']].copy()
fraud_data_temp_device.set_index('purchase_time', inplace=True)
fraud_data_temp_device = fraud_data_temp_device.sort_index()

device_transaction_counts = fraud_data_temp_device.groupby('device_id').rolling('24H')['purchase_value'].count().reset_index()
device_transaction_counts.rename(columns={'purchase_value': 'device_transactions_24hr'}, inplace=True)

device_transaction_values_sum = fraud_data_temp_device.groupby('device_id').rolling('24H')['purchase_value'].sum().reset_index()
device_transaction_values_sum.rename(columns={'purchase_value': 'device_value_24hr_sum'}, inplace=True)

fraud_data_df = pd.merge(fraud_data_df, device_transaction_counts, on=['device_id', 'purchase_time'], how='left')
fraud_data_df = pd.merge(fraud_data_df, device_transaction_values_sum, on=['device_id', 'purchase_time'], how='left')

# Drop the temporary datetime index column added by rolling for merging purposes
fraud_data_df.drop(columns=['purchase_time'], inplace=True)

# Also, consider frequency based on user_id and device_id combination if applicable
# For simplicity, we'll stick to user-level and device-level for now.

print("Fraud_Data.csv after feature engineering:")
print(fraud_data_df.head())
print(fraud_data_df.info())


# --- 3. Feature Engineering for creditcard.csv (Less extensive, mainly time-based insights if any) ---
print("\n--- Feature Engineering for Bank Data (creditcard.csv) ---")

# The 'Time' feature is already seconds from the first transaction.
# We can extract cyclical features from 'Time' if desired, but given it's already a numerical
# time elapsed feature and PCA components are present, simpler approaches might suffice.
# For now, we'll keep 'Time' as is. No new features are explicitly required here by the prompt
# beyond the existing Time, V1-V28, Amount.

print("No new explicit feature engineering applied for creditcard.csv based on prompt's details beyond existing Time, V_features, Amount.")
print(creditcard_df.head())
print(creditcard_df.info())

# Save engineered data
fraud_data_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_engineered.csv'), index=False)
creditcard_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_engineered.csv'), index=False) # Renamed for clarity
print(f"\nEngineered data saved to '{PROCESSED_DATA_PATH}'.")
print("--- Feature Engineering Complete ---")