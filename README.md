# Improved Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project aims to significantly enhance the detection of fraudulent activities in online e-commerce transactions and traditional bank credit card transactions for Adey Innovations Inc. By applying advanced data science techniques, including detailed data analysis, sophisticated feature engineering, robust machine learning model building, and powerful explainability methods (SHAP), we seek to:

*   **Accurately Identify Fraud:** Develop models that can precisely distinguish between legitimate and fraudulent transactions.
*   **Minimize Financial Losses:** Prevent monetary loss for Adey Innovations Inc. and its customers due to fraud.
*   **Build Customer Trust:** Enhance transaction security, fostering greater confidence among customers and financial institutions.
*   **Optimize Operational Efficiency:** Provide actionable insights for real-time monitoring and rapid response to suspicious activities.
*   **Balance Trade-offs:** Carefully manage the critical trade-off between security (minimizing false negatives/missing fraud) and user experience (minimizing false positives/flagging legitimate transactions).

## Business Need

In the financial technology sector, effective fraud detection is not just a technical challenge but a critical business imperative. The consequences of undetected fraud range from direct financial losses to severe reputational damage and erosion of customer trust. This project directly addresses these challenges by building intelligent systems capable of adapting to evolving fraud patterns, thereby safeguarding financial assets and reinforcing our commitment to security and customer satisfaction.

## Datasets

This project utilizes three distinct datasets, each presenting unique characteristics and challenges:

1.  **`Fraud_Data.csv` (E-commerce Transactions):**
    *   **Description:** Contains a diverse set of e-commerce transaction records.
    *   **Key Features:** `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, and the target `class` (1 for fraudulent, 0 for legitimate).
    *   **Critical Challenge:** Exhibits a high degree of class imbalance, where fraudulent transactions are a small minority.

2.  **`IpAddress_to_Country.csv` (IP to Country Mapping):**
    *   **Description:** A lookup table used to enrich `Fraud_Data.csv` with geographical information.
    *   **Key Features:** `lower_bound_ip_address`, `upper_bound_ip_address`, `country`.

3.  **`creditcard.csv` (Bank Credit Card Transactions):**
    *   **Description:** A dataset specifically curated for bank fraud detection, featuring anonymized transactional data.
    *   **Key Features:** `Time` (seconds elapsed from the first transaction), `V1` to `V28` (anonymized PCA components representing underlying transaction patterns), `Amount`, and the target `Class` (1 for fraudulent, 0 for legitimate).
    *   **Critical Challenge:** Presents an even more extreme class imbalance compared to the e-commerce dataset, typical of real-world financial fraud.

## Project Structure



## Project Structure
fraud_detection_week_8-9/
├── ── data/
│ ├── Fraud_Data.csv # Raw e-commerce transaction data
│ ├── IpAddress_to_Country.csv # IP address to country mapping data
│ └── creditcard.csv # Raw bank credit card transaction data
│ └── processed/ # Directory for intermediate processed data files
│ ├── fraud_data_processed.csv # E-commerce data after initial cleaning and IP merge
│ ├── fraud_data_engineered.csv # E-commerce data with engineered features
│ └── creditcard_processed.csv # Bank data after cleaning
│ └── creditcard_engineered.csv # Bank data (same as processed for this project)
├── notebooks/
│ ├── 1_EDA_Fraud_Data.ipynb # Exploratory Data Analysis for E-commerce data
│ ├── 2_EDA_CreditCard_Data.ipynb # Exploratory Data Analysis for Bank data
│ └── 3_Feature_Engineering_and_Modeling.ipynb # Prototyping for FE and model selection
├── src/ # Source code for the data science pipeline
│ ├── data_preprocessing.py # Handles data loading, cleaning, IP-to-country merge
│ ├── feature_engineering.py # Creates new time-based and velocity features
│ ├── model_training.py # Builds, trains, and evaluates ML models
│ ├── model_explainability.py # Generates SHAP plots for model interpretation
│ └── utils.py # Utility functions (e.g., feature type identification)
├── models/ # Stores trained machine learning models in .pkl format
│ ├── ecommerce_logistic_regression.pkl
│ ├── ecommerce_lightgbm.pkl
│ ├── bank_logistic_regression.pkl
│ ├── bank_lightgbm.pkl
│ ├── ecommerce_training_results.pkl # Stores test sets and best model for explainability
│ └── bank_training_results.pkl
├── reports/ # Stores project reports (PDF, Markdown) and generated visualizations
│ ├
│ ├── confusion_matrix_.png # Saved confusion matrix plots
│ ├── pr_curve_.png # Saved precision-recall curve plots
│ ├── shap_summary_dot_plot_.png # Saved SHAP summary dot plots
│ ├── shap_bar_plot_.png # Saved SHAP bar plots (feature importance)
│ └── shap_force_plot_instance_*.html # Saved interactive SHAP force plots for individual predictions
├── .gitignore # Specifies files/directories to be ignored by Git
├── requirements.txt # Lists all Python package dependencies and their versions
└── README.md # This comprehensive project overview


## Setup Instructions

To set up the environment and run the project, follow these steps:

1.  **Clone the repository:**

    git clone https://github.com/Fentahun022/fraud_detection_week_8-9.git
   
    Then, navigate into the cloned directory:

    cd fraud_detection_week_8-9


2.  **Place Data Files:**
    Ensure `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` are placed directly into the `data/` directory within your project structure.

3.  **Create a virtual environment (recommended):**
    This isolates your project's dependencies from your system-wide Python packages.

    python -m venv venv
   
    Activate the virtual environment:
    *   **On Windows:**

        .\venv\Scripts\activate
     
    *   **On macOS / Linux:**
 
        source venv/bin/activate
        
    (You should see `(venv)` appear at the beginning of your terminal prompt.)

4.  **Install required packages:**
    Once your virtual environment is active, install all dependencies listed in `requirements.txt`.

    pip install -r requirements.txt
    
    *If `requirements.txt` is not yet present, first install the necessary libraries and then generate it:*

    pip install pandas numpy scikit-learn matplotlib seaborn imblearn lightgbm shap joblib ipaddress notebook
    pip freeze > requirements.txt
    

## How to Run the Code

The project's full data processing and modeling pipeline can be executed by running the Python scripts in the `src/` directory sequentially.

**To run the entire pipeline:**

1.  **Ensure your virtual environment is activated** (as shown in Step 3 of "Setup Instructions").
2.  **Execute the scripts in the specified order:**
    ```bash
    echo "--- Running data_preprocessing.py (Step 1: Data Cleaning and IP Merge) ---"
    python src/data_preprocessing.py

    echo "--- Running feature_engineering.py (Step 2: Feature Engineering) ---"
    python src/feature_engineering.py

    echo "--- Running model_training.py (Step 3: Model Building, Training, and Evaluation) ---"
    python src/model_training.py

    echo "--- Running model_explainability.py (Step 4: Model Interpretation with SHAP) ---"
    python src/model_explainability.py

    echo "--- All pipeline scripts executed! Check 'data/processed/', 'models/', and 'reports/' for outputs. ---"
    ```

**To explore interactively with Jupyter Notebooks:**

Jupyter notebooks are provided for detailed Exploratory Data Analysis (EDA) and prototyping.
1.  **Ensure your virtual environment is activated.**
2.  **Launch Jupyter Notebook:**

    jupyter notebook
   
3.  Your web browser will open. Navigate to the `notebooks/` directory and open the `.ipynb` files to review the analysis and visualizations. You can run individual cells or the entire notebook (`Kernel -> Restart & Run All`).

