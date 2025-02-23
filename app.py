import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# GitHub repository details
GITHUB_USER = "rafaqatkhan-ai"
GITHUB_REPO = "learning-feedback"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/main"  # Point to the 'main' directory

# GitHub Personal Access Token (if repository is private)
GITHUB_TOKEN = None  # Replace with your token if needed

# Streamlit App Title
st.title("📚 🎓 Khan-AcadPredict 🎓 📚")

# Function to fetch CSV files from GitHub repository
def fetch_github_csv_files():
    def fetch_files_from_path(path):
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        response = requests.get(f"{GITHUB_API_URL}/{path}", headers=headers)
        if response.status_code == 200:
            files = response.json()
            csv_files = []
            for file in files:
                if file['type'] == 'file' and file['name'].endswith('.csv'):
                    csv_files.append(file['download_url'])
                elif file['type'] == 'dir':
                    csv_files.extend(fetch_files_from_path(file['path']))
            return csv_files
        else:
            st.error(f"Failed to fetch files from GitHub: {response.status_code}")
            return []

    return fetch_files_from_path("")  # Start from the root of the 'main' directory

# Load dataset (either from GitHub or uploaded file)
st.sidebar.header("Select Dataset")

# Fetch available datasets from GitHub
github_csv_files = fetch_github_csv_files()

# Add dropdown for GitHub datasets
selected_github_csv = st.sidebar.selectbox("Choose a dataset from GitHub:", ["None"] + github_csv_files)

# File upload option
uploaded_file = st.sidebar.file_uploader("Or Upload a CSV file", type=["csv"])

# Initialize session state for dataset storage
if "df" not in st.session_state:
    st.session_state.df = None

# Load selected GitHub dataset
if selected_github_csv != "None":
    st.session_state.df = pd.read_csv(selected_github_csv)
    st.write(f"Loaded dataset from GitHub: {selected_github_csv}")

# Load uploaded dataset
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset loaded successfully!")

# Check if a dataset is loaded
if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write("Dataset shape:", df.shape)
    st.write("Dataset columns:", df.columns)

    # Function to preprocess data
    def preprocess_data(df):
        st.write("Preprocessing data...")
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target

        st.write("Features shape:", X.shape)
        st.write("Target shape:", y.shape)
        st.write("Unique target values:", np.unique(y))

        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(exclude=['object']).columns

        st.write("Categorical columns:", cat_cols)
        st.write("Numerical columns:", num_cols)

        # Standardize numerical features
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Encode categorical features
        if len(cat_cols) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
            X_cat.columns = encoder.get_feature_names_out(cat_cols)
            X = X.drop(columns=cat_cols).reset_index(drop=True)
            X = pd.concat([X, X_cat], axis=1)

        st.write("Processed features shape:", X.shape)
        return X, y

    # Function to train multiple models
    def train_models(X_train, X_test, y_train, y_test):
        st.write("Training models...")
        classifiers = {
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0)
        }
        results = {}

        for name, clf in classifiers.items():
            st.write(f"Training {name}...")
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                results[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1 Score": f1_score(y_test, y_pred, average='weighted')
                }
            except Exception as e:
                st.error(f"Error training {name}: {e}")
                results[name] = {"Error": str(e)}

        return results

    # Function to train a Deep Neural Network
    def train_dnn(X_train, X_test, y_train, y_test):
        st.write("Training Deep Neural Network...")
        try:
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
        except Exception as e:
            st.error(f"Error training DNN: {e}")
            return {"Error": str(e)}

    # Button to trigger training
    if st.button("Train Models"):
        st.write("### Training in Progress... ⏳")
        
        try:
            # Preprocess data
            X, y = preprocess_data(df)

            # Handle class imbalance using SMOTE
            if len(np.unique(y)) > 1:
                smote = SMOTE()
                X_resampled, y_resampled = smote.fit_resample(X, y)
            else:
                st.warning("Only one class in the target variable. Skipping SMOTE.")
                X_resampled, y_resampled = X, y

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Train machine learning models
            model_results = train_models(X_train, X_test, y_train, y_test)

            # Display results
            st.subheader("Evaluation Results for Machine Learning Models")
            for model, metrics in model_results.items():
                st.write(f"**{model}**")
                st.write(metrics)

            # Train and evaluate Deep Neural Network
            st.write("Training deep learning model, please wait... ⏳")
            dnn_results = train_dnn(X_train, X_test, y_train, y_test)

            st.subheader("Evaluation Results for Deep Neural Network")
            st.write(dnn_results)

            st.success("🎉 Training Completed Successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
