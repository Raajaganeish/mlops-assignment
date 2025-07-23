import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_housing():
    df = pd.read_csv("data/raw/housing.csv")
 
    # Check for missing values
    df.dropna(inplace=True)
 
    # Feature scaling (StandardScaler)
    features = df.drop(columns=["MedHouseVal"])
    target = df["MedHouseVal"]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = pd.DataFrame(X_scaled, columns=features.columns)
 
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.2, random_state=42
    )
 
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X_train.join(y_train).to_csv("data/processed/train.csv", index=False)
    X_test.join(y_test).to_csv("data/processed/test.csv", index=False)
    print("Data preprocessed and saved to data/processed/train.csv and data/processed/test.csv")

preprocess_housing()