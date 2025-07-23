import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
 
# Load data
df = pd.read_csv("data/processed/housing_processed.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define experiments
mlflow.set_experiment("CaliforniaHousingPrediction")
 
best_model = None
best_rmse = float("inf")
 
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}
 
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
 
        # Log params, metrics and model
        if name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model", registered_model_name="HousingModel")
 
        print(f"{name} RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model