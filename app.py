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

st.title("EduPredict 🎓 - Student Performance Predictor")

# GitHub Repository where CSV files are stored
GITHUB_REPO = "https://github.com/yourusername/your-repo-name"  # <-- Change this
GITHUB_RAW = "https://raw.githubusercontent.com/yourusername/your-repo-name/main/"  # <-- Change this

# Function to get available datasets from GitHub
@st.cache_data
def get_available_datasets():
    return ["student_data1.csv", "student_data2.csv", "student_data3.csv"]  # <-- Update with actual CSV file names

# Function to load CSV from GitHub
def load_data_from_github(file_name):
    url = GITHUB_RAW + file_name
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load dataset from GitHub.")
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

# Function to train DNN model
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

# Sidebar: Choose dataset source
st.sidebar.header("Select or Upload Dataset")
dataset_choice = st.sidebar.selectbox("Choose a dataset:", ["Select from GitHub", "Upload a file"])

if dataset_choice == "Select from GitHub":
    dataset_list = get_available_datasets()
    selected_dataset = st.sidebar.selectbox("Available Datasets:", dataset_list)

    if selected_dataset:
        df = load_data_from_github(selected_dataset)
        if df is not None:
            st.success(f"Loaded dataset: {selected_dataset}")
elif dataset_choice == "Upload a file":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded custom dataset!")

# Process the dataset if loaded
if 'df' in locals():
    X, y = preprocess_data(df)

    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train ML models
    st.write("Training ML models, please wait...")
    model_results = train_models(X_train, X_test, y_train, y_test)

    st.subheader("Evaluation Results for ML Models")
    for model, metrics in model_results.items():
        st.write(f"**{model}**")
        st.write(metrics)

    # Train DNN
    st.write("Training Deep Learning Model, please wait...")
    dnn_results = train_dnn(X_train, X_test, y_train, y_test)

    st.subheader("Evaluation Results for DNN Model")
    st.write(dnn_results)

    st.success("Training & Evaluation Completed! 🎉")
