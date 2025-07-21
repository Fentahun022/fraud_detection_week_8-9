# Improved Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project aims to enhance fraud detection capabilities for Adey Innovations Inc., a leading financial technology company. We develop and evaluate robust machine learning models to identify fraudulent activities in both e-commerce and bank credit transactions. A core focus is on addressing the critical challenge of class imbalance inherent in fraud datasets and balancing the trade-off between security and user experience (minimizing false positives while detecting true fraud).

The project leverages detailed data analysis, advanced feature engineering (including geolocation and time-based patterns), modern machine learning models (Logistic Regression, Gradient Boosting), and explainability techniques (SHAP) to provide actionable insights into fraud drivers.

## Business Need

Effective fraud detection is crucial for preventing financial losses and building trust with customers and financial institutions. This project directly addresses this need by:
*   **Minimizing Financial Losses:** Accurately identifying fraudulent transactions reduces direct monetary loss.
*   **Enhancing Customer Trust:** Robust security measures instill confidence in users and partners.
*   **Optimizing Operations:** Streamlining real-time monitoring and reporting for quicker response to suspicious activities.
*   **Balancing Security and UX:** Carefully managing false positives to avoid alienating legitimate customers.

## Datasets

This project utilizes three datasets:

1.  **`Fraud_Data.csv` (E-commerce Transactions):**
    *   `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class` (target: 1=fraud, 0=legitimate).
    *   **Critical Challenge:** High class imbalance.

2.  **`IpAddress_to_Country.csv` (IP to Country Mapping):**
    *   `lower_bound_ip_address`, `upper_bound_ip_address`, `country`.
    *   Used to enrich `Fraud_Data.csv` with geographical information.

3.  **`creditcard.csv` (Bank Credit Card Transactions):**
    *   `Time`, `V1` to `V28` (anonymized PCA components), `Amount`, `Class` (target: 1=fraud, 0=legitimate).
    *   **Critical Challenge:** Extreme class imbalance.

## Project Structure
fraud_detection_week_8-9/
├── data/
│ ├── Fraud_Data.csv
│ ├── IpAddress_to_Country.csv
│ └── creditcard.csv
│ └── processed/ # Stores intermediate processed/engineered data
│ ├── fraud_data_processed.csv
│ ├── fraud_data_engineered.csv
│ └── creditcard_processed.csv
├── notebooks/
│ ├── 1_EDA_Fraud_Data.ipynb
│ ├── 2_EDA_CreditCard_Data.ipynb
│ └── 3_Feature_Engineering_and_Modeling.ipynb
├── src/
│ ├── data_preprocessing.py # Handles data loading, cleaning, IP-to-country merge
│ ├── feature_engineering.py # Creates new time-based and velocity features
│ ├── model_training.py # Builds, trains, and evaluates ML models
│ ├── model_explainability.py # Generates SHAP plots for model interpretation
│ └── utils.py # Optional: For shared helper functions
├── models/
│ ├── ecommerce_logistic_regression.pkl # Saved trained models
│ ├── ecommerce_lightgbm.pkl
│ ├── bank_logistic_regression.pkl
│ └── bank_lightgbm.pkl
├── reports/
│ ├── interim_1_report.pdf # Project reports (PDF or Markdown)
│ ├── interim_2_report.pdf
│ ├── final_report_or_blog_post.pdf/md
│ ├── confusion_matrix_E_commerce_Data_Logistic_Regression.png # Saved plots
│ ├── pr_curve_E_commerce_Data_Logistic_Regression.png
│ └── shap_summary_plot_E_commerce_Data_LightGBM.png
├── .gitignore # Specifies files/directories to ignore in Git
├── requirements.txt # Lists all Python package dependencies
└── README.md


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

