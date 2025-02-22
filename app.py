import streamlit as st
import pandas as pd
import numpy as np
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
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM, SimpleRNN, Flatten
from tensorflow.keras.optimizers import Adam

# Load Data
@st.cache_data
def load_data():
    file_path = "student_prediction.csv"
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values.astype(np.float32)  # Features
    y = df.iloc[:, -1].values.astype(np.int64)  # Target
    return X, y

# Train and evaluate models
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
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    
    return results

# Train Neural Network Models
def train_neural_networks(X_train, X_test, y_train, y_test):
    models = {
        'DNN': Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
    }

    results = {}

    for name, model in models.items():
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    
    return results

# Streamlit App
st.title("Student Performance Prediction App 🎓")

# Load Data
X, y = load_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Model Training and Evaluation
if st.button("Train Models"):
    st.write("Training models, please wait...")
    model_results = train_models(X_train, X_test, y_train, y_test)

    st.subheader("Model Evaluation Results")
    for model, metrics in model_results.items():
        st.write(f"**{model}**")
        st.write(metrics)

    nn_results = train_neural_networks(X_train, X_test, y_train, y_test)
    st.subheader("Neural Network Performance")
    for model, metrics in nn_results.items():
        st.write(f"**{model}**")
        st.write(metrics)

st.sidebar.header("Upload Student Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for prediction", type=["csv"])
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)

    # Ensure uploaded data has same features as training data
    expected_columns = X_train.shape[1]  # Number of features used in training

    if user_data.shape[1] != expected_columns:
        st.error(f"Uploaded file has {user_data.shape[1]} features, but expected {expected_columns}. Please check your file.")
    else:
        user_data = scaler.transform(user_data)  # Transform with the same scaler

        # Load trained model and predict
        model = XGBClassifier()
        model.fit(X_train, y_train)  # Train model (consider saving/loading model instead)
        prediction = model.predict(user_data)

        st.write("Predicted Class:", prediction)

