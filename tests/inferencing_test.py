from fastapi.testclient import TestClient
from api.inferencing import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Housing Price Prediction API"}


def test_predict_valid_input():
    sample_input = {
        "MedInc": 8.3,
        "HouseAge": 21.0,
        "AveRooms": 6.1,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)


def test_predict_missing_field():
    # Missing 'AveRooms'
    invalid_input = {
        "MedInc": 8.3,
        "HouseAge": 21.0,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422
    assert "error" in response.json() or "detail" in response.json()


def test_predict_invalid_type():
    # 'HouseAge' is a string instead of float
    invalid_input = {
        "MedInc": 8.3,
        "HouseAge": "twenty-one",
        "AveRooms": 6.1,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422
    assert "error" in response.json() or "detail" in response.json()


def test_predict_negative_income():
    invalid_input = {
        "MedInc": -5.0,  # Should fail due to gt=0 constraint
        "HouseAge": 21.0,
        "AveRooms": 6.1,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422
    assert "error" in response.json() or "detail" in response.json()


def test_predict_invalid_latitude():
    invalid_input = {
        "MedInc": 8.3,
        "HouseAge": 21.0,
        "AveRooms": 6.1,
        "AveBedrms": 1.0,
        "Population": 1400.0,
        "AveOccup": 3.2,
        "Latitude": 123.45,  # Out of valid latitude range (-90 to 90)
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422
    assert "error" in response.json() or "detail" in response.json()
