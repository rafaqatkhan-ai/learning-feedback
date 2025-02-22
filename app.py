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

    # Separate features (X_user) and target (y_user)
    X_user = user_data.iloc[:, :-1]  # All columns except last one
    y_user = user_data.iloc[:, -1]   # Last column as target (not used in prediction)

    # Identify categorical and numerical columns
    cat_cols = X_user.select_dtypes(include=['object']).columns  # Categorical columns
    num_cols = X_user.select_dtypes(exclude=['object']).columns  # Numerical columns

    # Apply transformations
    if len(num_cols) > 0:
        X_user[num_cols] = scaler.transform(X_user[num_cols])  # Standardize numerical data

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_user_cat = pd.DataFrame(encoder.fit_transform(X_user[cat_cols]))  # One-hot encode
        X_user_cat.columns = encoder.get_feature_names_out(cat_cols)  # Keep column names
        X_user = X_user.drop(columns=cat_cols).reset_index(drop=True)  # Drop original categorical columns
        X_user = pd.concat([X_user, X_user_cat], axis=1)  # Merge with encoded columns

    # Ensure all columns match the trained model
    missing_cols = set(X_train.shape[1]) - set(X_user.shape[1])
    for col in missing_cols:
        X_user[col] = 0  # Add missing columns with 0s
    X_user = X_user[X_train.columns]  # Ensure correct column order

    # Train the model (consider saving/loading the trained model instead)
    model = XGBClassifier()
    model.fit(X_train, y_train)  # Retraining the model

    # Predict using the model
    predictions = model.predict(X_user)

    st.write("Predicted Class:", predictions)
