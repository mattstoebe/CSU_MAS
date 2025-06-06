import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
from dotenv import load_dotenv
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load environment variables
load_dotenv()

# Get configuration from environment variables
APP_NAME = os.getenv('APP_NAME', 'TBI Prediction Model')
MODEL_PATH = os.getenv('MODEL_PATH', 'model.joblib')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10485760))  # Default 10MB
ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'csv,xlsx').split(',')
MAX_ROWS_PREVIEW = int(os.getenv('MAX_ROWS_PREVIEW', 5))
PLOT_WIDTH = int(os.getenv('PLOT_FIGURE_SIZE_WIDTH', 10))
PLOT_HEIGHT = int(os.getenv('PLOT_FIGURE_SIZE_HEIGHT', 6))

# Define feature treatments
feature_treatments = {
    'age': 'continuous',
    'race7': 'factor_nominal',
    'ethnic3': 'factor_nominal',
    'sex2': 'factor_nominal',
    'sig_other': 'factor_nominal',
    'tobacco': 'factor_nominal',
    'alcohol': 'factor_nominal',
    'drugs': 'factor_nominal',
    'MedianIncomeForZip': 'continuous',
    'PercentAboveHighSchoolEducationForZip': 'continuous',
    'PercentAboveBachelorsEducationForZip': 'continuous',
    'payertype': 'factor_nominal',
    'tbiS02': 'continuous',
    'tbiS06': 'continuous',
    'tbiS09': 'continuous',
    'ptp1_yn': 'factor_nominal',
    'ptp2_yn': 'factor_nominal',
    'ptp0_yn': 'factor_nominal',
    'ed_yn': 'factor_nominal',
    'icu': 'factor_nominal',
    'delirium': 'factor_nominal',
    'agitated': 'factor_nominal',
    'lethargic': 'factor_nominal',
    'comatose': 'factor_nominal',
    'disoriented': 'factor_nominal',
    'gcs_min': 'continuous',
    'gcs_max': 'continuous',
    'adl_min': 'continuous',
    'adl_max': 'continuous',
    'mobility_min': 'continuous',
    'mobility_max': 'continuous',
    'los_total': 'continuous',
    'dc_setting': 'factor_nominal',
    'prehosp': 'factor_nominal',
    'posthosp': 'factor_nominal',
}

# Set page config
st.set_page_config(page_title=APP_NAME, layout="wide")

# Title and description
st.title(APP_NAME)
st.write("Upload your CSV file to get predictions and SHAP value analysis.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=ALLOWED_EXTENSIONS)

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE/1048576:.1f}MB")
    else:
        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(MAX_ROWS_PREVIEW))
            
            # Load the model
            try:
                model = joblib.load(MODEL_PATH)
                
                # Preprocess the data
                nominal_cols = [col for col, treatment in feature_treatments.items() if treatment == 'factor_nominal']
                continuous_cols = [col for col, treatment in feature_treatments.items() if treatment == 'continuous']
                
                # One-hot encode categorical variables
                df_processed = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
                
                # Make predictions
                predictions = model.predict_proba(df_processed)[:, 1]
                
                # Display predictions
                st.subheader("Predictions")
                predictions_df = pd.DataFrame({
                    'Row': range(1, len(predictions) + 1),
                    'Probability': predictions,
                    'Prediction': (predictions > 0.5).astype(int)  # Using 0.5 as default threshold
                })
                st.dataframe(predictions_df)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_processed)
                
                # Plot SHAP summary
                st.subheader("SHAP Value Analysis")
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                shap.summary_plot(shap_values, df_processed, plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Download predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except FileNotFoundError:
                st.error(f"Model file not found. Please ensure '{MODEL_PATH}' exists in the correct location.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}") 