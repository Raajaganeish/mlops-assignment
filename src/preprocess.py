import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_housing():
    df = pd.read_csv("data/raw/housing.csv")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate features and target
    features = df.drop(columns=["MedHouseVal"])
    target = df["MedHouseVal"]

    # Fit and apply scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = pd.DataFrame(X_scaled, columns=features.columns)

    # Combine scaled features with target
    processed_df = X_scaled.join(target)

    # Save preprocessed data
    os.makedirs("data/processed", exist_ok=True)
    processed_df.to_csv("data/processed/housing_processed.csv", index=False)
    print("✅ Preprocessed data saved to data/processed/housing_processed.csv")

    # ✅ Save the fitted scaler for inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    print("✅ Test logger to trigger re-train pipeline")


# Run the function
preprocess_housing()
