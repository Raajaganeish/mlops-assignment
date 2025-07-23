import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_housing():
    df = pd.read_csv("data/raw/housing.csv")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Feature scaling (StandardScaler)
    features = df.drop(columns=["MedHouseVal"])
    target = df["MedHouseVal"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = pd.DataFrame(X_scaled, columns=features.columns)

    # Combine scaled features with target
    processed_df = X_scaled.join(target)

    # Save the preprocessed data
    os.makedirs("data/processed", exist_ok=True)
    processed_df.to_csv("data/processed/housing_processed.csv", index=False)
    print("Data preprocessed and saved to data/processed/housing_processed.csv")

preprocess_housing()