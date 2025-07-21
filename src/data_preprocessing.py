
import pandas as pd
import numpy as np
import ipaddress
import os
import sys

# Define paths
DATA_PATH = 'data/'
PROCESSED_DATA_PATH = 'data/processed/'

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("--- Starting Data Preprocessing ---")

# --- 1. Load Datasets ---
print("Loading datasets...")
try:
    fraud_data_df = pd.read_csv(os.path.join(DATA_PATH, 'Fraud_Data.csv'))
    ip_country_df = pd.read_csv(os.path.join(DATA_PATH, 'IpAddress_to_Country.csv'))
    creditcard_df = pd.read_csv(os.path.join(DATA_PATH, 'creditcard .csv'))
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the CSV files are in the '{DATA_PATH}' directory.")
    sys.exit(1)

# --- 2. Initial Cleaning for Fraud_Data.csv ---
print("\n--- Processing Fraud_Data.csv ---")
print("Missing values before handling:\n", fraud_data_df.isnull().sum())
initial_rows_fraud = fraud_data_df.shape[0]
fraud_data_df.dropna(inplace=True)
print(f"Dropped {initial_rows_fraud - fraud_data_df.shape[0]} rows with missing values.")

print("Converting time columns to datetime...")
fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])
fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])
print("Data types after conversion for Fraud_Data.csv:\n", fraud_data_df[['signup_time', 'purchase_time']].info())

print("Checking for duplicates...")
initial_rows_fraud_after_na = fraud_data_df.shape[0]
fraud_data_df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_fraud_after_na - fraud_data_df.shape[0]} duplicate rows.")
print(f"Fraud_Data.csv shape after initial cleaning: {fraud_data_df.shape}")

# --- 3. Initial Cleaning for creditcard.csv ---
print("\n--- Processing creditcard.csv ---")
print("Missing values before handling:\n", creditcard_df.isnull().sum())
initial_rows_creditcard = creditcard_df.shape[0]
creditcard_df.dropna(inplace=True)
print(f"Dropped {initial_rows_creditcard - creditcard_df.shape[0]} rows with missing values.")

print("Checking for duplicates...")
initial_rows_creditcard_after_na = creditcard_df.shape[0]
creditcard_df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_creditcard_after_na - creditcard_df.shape[0]} duplicate rows.")
print(f"creditcard.csv shape after initial cleaning: {creditcard_df.shape}")

# --- 4. Initial Cleaning for IpAddress_to_Country.csv ---
print("\n--- Processing IpAddress_to_Country.csv ---")
print("Missing values before handling:\n", ip_country_df.isnull().sum())
initial_rows_ip_country = ip_country_df.shape[0]
ip_country_df.dropna(inplace=True)
print(f"Dropped {initial_rows_ip_country - ip_country_df.shape[0]} rows with missing values.")

print("Checking for duplicates...")
initial_rows_ip_country_after_na = ip_country_df.shape[0]
ip_country_df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows_ip_country_after_na - ip_country_df.shape[0]} duplicate rows.")
print(f"IpAddress_to_Country.csv shape after initial cleaning: {ip_country_df.shape}")

# --- 5. Merge Datasets for Geolocation Analysis (Fraud_Data & IpAddress_to_Country) ---
print("\n--- Merging Fraud_Data with IpAddress_to_Country for Geolocation ---")

# --- MODIFIED: New IP conversion function for numerical IPs ---
def numeric_ip_to_int(ip_val):
    """Converts a numerical IP address representation (float/int) to an integer."""
    if pd.isna(ip_val):
        return np.nan
    try:
        return int(ip_val)
    except (ValueError, TypeError): # Catch if it's not a valid number
        return np.nan
# --- END MODIFIED ---

# Convert IP addresses in Fraud_Data to integer
print("Converting 'ip_address' (float) to integer in Fraud_Data.csv...")
fraud_data_df['ip_address_int'] = fraud_data_df['ip_address'].apply(numeric_ip_to_int)
initial_rows_fraud_ip_int = fraud_data_df.shape[0]
fraud_data_df.dropna(subset=['ip_address_int'], inplace=True)
print(f"Dropped {initial_rows_fraud_ip_int - fraud_data_df.shape[0]} rows due to invalid IP address conversion.")
# Ensure it's integer type
if not fraud_data_df.empty:
    fraud_data_df['ip_address_int'] = fraud_data_df['ip_address_int'].astype(int)

# Convert IP ranges in IpAddress_to_Country to integer
print("Converting IP range columns (float) to integer in IpAddress_to_Country.csv...")
ip_country_df['lower_bound_ip_address_int'] = ip_country_df['lower_bound_ip_address'].apply(numeric_ip_to_int)
ip_country_df['upper_bound_ip_address_int'] = ip_country_df['upper_bound_ip_address'].apply(numeric_ip_to_int)
initial_rows_ip_country_int = ip_country_df.shape[0]
ip_country_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)
print(f"Dropped {initial_rows_ip_country_int - ip_country_df.shape[0]} rows due to invalid IP range conversion.")
# Ensure they are integer types
if not ip_country_df.empty:
    ip_country_df['lower_bound_ip_address_int'] = ip_country_df['lower_bound_ip_address_int'].astype(int)
    ip_country_df['upper_bound_ip_address_int'] = ip_country_df['upper_bound_ip_address_int'].astype(int)


# Sorting ip_country_df for the merge operation
ip_country_df_sorted = ip_country_df.sort_values(by='lower_bound_ip_address_int').reset_index(drop=True)

# Function to find country for an IP
# This function remains the same as its logic for finding the country in the sorted range is correct.
def get_country_from_ip(ip_int, ip_country_df_sorted):
    mask = (ip_int >= ip_country_df_sorted['lower_bound_ip_address_int']) & \
           (ip_int <= ip_country_df_sorted['upper_bound_ip_address_int'])
    
    matching_countries = ip_country_df_sorted.loc[mask, 'country']
    
    if not matching_countries.empty:
        return matching_countries.iloc[0]
    return np.nan

print("Mapping IP addresses to countries (this may take some time)...")
# Only attempt to merge if fraud_data_df is not empty after previous IP conversion
if not fraud_data_df.empty:
    fraud_data_df['country'] = fraud_data_df['ip_address_int'].apply(lambda x: get_country_from_ip(x, ip_country_df_sorted))

    initial_rows_after_country_map = fraud_data_df.shape[0]
    fraud_data_df.dropna(subset=['country'], inplace=True)
    print(f"Dropped {initial_rows_after_country_map - fraud_data_df.shape[0]} rows where country could not be mapped.")
else:
    print("Skipping country mapping as fraud_data_df is already empty.")


# --- Convert target columns to categorical type explicitly ---
# This is crucial for Seaborn to handle the 'hue' parameter correctly with discrete values,
# and for proper model handling later.
print("\nConverting target 'class' and 'Class' columns to categorical dtype...")
# Only apply if the dataframe is not empty and the column exists
if not fraud_data_df.empty and 'class' in fraud_data_df.columns:
    fraud_data_df['class'] = fraud_data_df['class'].astype('category')
if not creditcard_df.empty and 'Class' in creditcard_df.columns:
    creditcard_df['Class'] = creditcard_df['Class'].astype('category')


print("\nFraud_Data.csv after initial preprocessing and country merge:")
if not fraud_data_df.empty:
    print(fraud_data_df.head())
    fraud_data_df.info()
else:
    print("Fraud_Data.csv is empty.")


# Save processed data
if not fraud_data_df.empty:
    fraud_data_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'fraud_data_processed.csv'), index=False)
    print(f"\nProcessed fraud data saved to '{PROCESSED_DATA_PATH}'.")
else:
    print("\nSkipping saving fraud_data_processed.csv as DataFrame is empty.")

if not creditcard_df.empty:
    creditcard_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_processed.csv'), index=False)
    print(f"Processed credit card data saved to '{PROCESSED_DATA_PATH}'.")
else:
    print("\nSkipping saving creditcard_processed.csv as DataFrame is empty.")

print("--- Data Preprocessing Complete ---")