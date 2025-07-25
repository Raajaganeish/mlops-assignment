from fastapi.testclient import TestClient
from api.inferencing import app
 
client = TestClient(app)
 
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Housing Price Prediction API"}
 
def test_predict_endpoint():
    sample_input = {
        "MedInc": 8.3,
        "HouseAge": 21.0,
        "AveRooms": 6.1,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
 
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)