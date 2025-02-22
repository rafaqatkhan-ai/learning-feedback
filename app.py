import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.title("📚 🎓 Khan-AcadPredict 🎓 📚")

# GitHub Repository Information
GITHUB_USER = "rafaqatkhan-ai"
GITHUB_REPO = "learning-feedback"
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/"

# Function to fetch CSV files from GitHub repository
@st.cache_data
def get_github_csv_files():
    repo_api = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/"
    response = requests.get(repo_api)

    if response.status_code == 200:
        files = response.json()
        csv_files = [file['name'] for file in files if file['name'].endswith('.csv')]
        return csv_files
    else:
        st.error("Failed to fetch CSV files from GitHub.")
        return []

# Function to load CSV from GitHub
def load_data_from_github(file_name):
    url = GITHUB_RAW + file_name
    response = requests.get(url)
    
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error(f"Failed to load dataset: {file_name} from GitHub.")
        return None

# Function to preprocess data
def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
        X_cat.columns = encoder.get_feature_names_out(cat_cols)
        X = X.drop(columns=cat_cols).reset_index(drop=True)
        X = pd.concat([X, X_cat], axis=1)

    return X, y

# Function to train ML models
def train_models(X_train, X_test, y_train, y_test):
    classifiers = {
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(),
        'LightGBM': lgb.LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }
    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        }

    return results

# Function to train Deep Neural Network
def train_dnn(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

# Sidebar: Dataset selection
st.sidebar.header("Step 1: Select or Upload Dataset")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Select from GitHub", "Upload a file"])

df = None  # Initialize empty dataframe

if dataset_source == "Select from GitHub":
    csv_files = get_github_csv_files()
    selected_file = st.sidebar.selectbox("Available Datasets:", csv_files)

elif dataset_source == "Upload a file":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Button to confirm dataset selection
if st.sidebar.button("Load Dataset"):
    if dataset_source == "Select from GitHub" and selected_file:
        df = load_data_from_github(selected_file)
        st.success(f"Loaded dataset: {selected_file}")
    elif dataset_source == "Upload a file" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded custom dataset!")
    else:
        st.warning("Please select a dataset first.")

# Training section (only appears if dataset is loaded)
if df is not None:
    st.sidebar.header("Step 2: Train Models")
    if st.sidebar.button("Start Training"):
        X, y = preprocess_data(df)

        # Handle class imbalance using SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train ML models
        st.write("Training machine learning models, please wait...")
        model_results = train_models(X_train, X_test, y_train, y_test)

        st.subheader("Evaluation Results for Machine Learning Models")
        for model, metrics in model_results.items():
            st.write(f"**{model}**")
            st.write(metrics)

        # Train and evaluate DNN
        st.write("Training deep learning model, please wait...")
        dnn_results = train_dnn(X_train, X_test, y_train, y_test)

        st.subheader("Evaluation Results for Deep Neural Network")
        st.write(dnn_results)

        st.success("Training & Evaluation Completed! 🎉")
