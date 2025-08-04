import datetime
import os
import warnings

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# --------------------------
# Step 1: Load preprocessed data
# --------------------------
def load_data(path="data/processed/housing_processed.csv"):
    df = pd.read_csv(path)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y


# --------------------------
# Step 2: Scale features
# --------------------------
def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")

    return X_train_scaled, X_val_scaled, scaler


# --------------------------
# Step 3: Train and evaluate models
# --------------------------
def train_models(X_train, y_train, X_val, y_val):
    mlflow.set_experiment("CaliforniaHousingPrediction")

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
    }

    best_model = None
    best_rmse = float("inf")
    best_model_name = ""
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            if name == "DecisionTree":
                mlflow.log_param("max_depth", 5)
            mlflow.log_metric("rmse", rmse)

            print(f"{name} RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name
                best_run_id = run.info.run_id

            print(f"Best model so far: {best_model_name} with RMSE: {best_rmse:.4f}")

    return best_model, best_model_name, best_run_id, X_val


# --------------------------
# Step 4: Log and register model, return version
# --------------------------
def log_and_register_model(model, run_id, X_val_scaled):
    input_example = X_val_scaled.iloc[:1]
    signature = infer_signature(X_val_scaled, model.predict(input_example))

    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        model_uri = f"runs:/{run_id}/model"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`artifact_path` is deprecated")
            mlflow.register_model(model_uri, "BestCaliforniaHousingModel")

        print("✅ Model registered to MLflow")

    client = MlflowClient()
    latest_versions = client.get_latest_versions("BestCaliforniaHousingModel")
    version = latest_versions[0].version
    return str(version)


# --------------------------
# Step 5: Save model to disk with metadata
# --------------------------
def save_model_locally(model, model_version, path="models/best_model.pkl"):
    model_package = {
        "model": model,
        "model_type": type(model).__name__,
        "model_version": model_version,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    joblib.dump(model_package, path)
    print(f"✅ Best model saved with metadata to {path}")


# --------------------------
# Pipeline Runner
# --------------------------
def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_scaled, X_val_scaled, _ = scale_features(X_train, X_val)
    best_model, best_model_name, best_run_id, X_val = train_models(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    model_version = log_and_register_model(best_model, best_run_id, X_val_scaled)
    save_model_locally(best_model, model_version)

    print(
        f"✅ Training completed: model = {best_model_name}, "
        f"version = {model_version}"
    )
    print("✅ Test 2 logger to trigger re-train pipeline")


# Run
if __name__ == "__main__":
    main()
