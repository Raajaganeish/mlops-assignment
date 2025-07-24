import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib
import os
 
# Load data
df = pd.read_csv("data/processed/housing_processed.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define experiments
mlflow.set_experiment("CaliforniaHousingPrediction")
 
best_model = None
best_rmse = float("inf")
best_model_name = ""
best_run_id = None

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        # Log params, metrics and model
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

print("Training complete.")

# Register the best model
with mlflow.start_run(run_id=best_run_id):
    mlflow.sklearn.log_model(best_model, "model", registered_model_name="BestCaliforniaHousingModel")
    print(f"Best model '{best_model_name}' registered to MLflow.")


 
# Save the best model as best_model.pkl
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
print("Best model saved to models/best_model.pkl")