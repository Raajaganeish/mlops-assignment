# MLOps Assignment: End-to-End Pipeline for ML Model Lifecycle

## 🎯 Objective

This project demonstrates a **complete MLOps pipeline** for training, tracking, packaging, deploying, and monitoring a machine learning model using best practices. It is developed as part of the assignment:  
**“Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices”**

We chose the **California Housing dataset** (regression) to build this pipeline.

---

## 🧠 Problem Statement

The goal is to predict median house values in California districts using input features like population, median income, and housing characteristics. The pipeline ensures reproducibility, modularity, and operationalization of the model.

---

## 📦 Features

✅ Load and preprocess data  
✅ Track datasets and model versions using DVC and MLflow  
✅ Train and evaluate multiple models  
✅ Package the model as a REST API using FastAPI  
✅ Containerize the API with Docker  
✅ Set up CI/CD with GitHub Actions  
✅ Log prediction requests and monitor usage  
✅ Optionally expose metrics for monitoring  

---

## 📂 Project Structure

```
├── data/
│   ├── raw/                 # Raw data (e.g., housing.csv)
│   └── processed/           # Processed features and target
├── src/
│   ├── load_data.py         # Script to download and save raw data
│   ├── preprocess.py        # Data cleaning and feature scaling
├── api/                     # FastAPI app code
├── models/                  # Saved models and artifacts
├── logs/                    # Logs from prediction requests
├── .github/
│   └── workflows/           # GitHub Actions config
├── dvc.yaml                 # DVC pipeline file
├── README.md                # Project overview and setup
└── requirements.txt         # Python dependencies
```

---

## 🚀 Technologies Used

- **MLflow** – Experiment tracking and model registry  
- **DVC** – Data and model versioning  
- **FastAPI** – REST API to serve the model  
- **Docker** – Containerization of the app  
- **GitHub Actions** – CI/CD automation  
- **Logging module** – Store logs for prediction history  
- **Optional**: Prometheus/Grafana for metrics  

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mlops-housing-assignment.git
cd mlops-housing-assignment
```

### 2. Reproduce data and model
```bash
dvc pull
python src/train.py
```

### 3. Run the API locally
```bash
uvicorn api.main:app --reload
```

### 4. Build & run Docker container
```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

### 5. Test the API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"      -d '{"feature_1": 3.5, "feature_2": 45.2, ...}'
```

---

## 🔁 CI/CD with GitHub Actions

Every push triggers a GitHub Actions workflow to:

- Lint and test the code
- Build Docker image
- Optionally deploy it (locally or to EC2 using script)

---

## 📊 Monitoring & Logging

- Every prediction request is logged into `logs/` folder.
- Optional `/metrics` endpoint can be added to expose Prometheus metrics.
- Logs can be used to trigger retraining based on data drift.

---

## ✅ Assignment Deliverables

| Task                                  | Status   |
|---------------------------------------|----------|
| Git & DVC Setup                       | ✅ Done |
| Model Training & MLflow Tracking      | ✅ Done |
| REST API with FastAPI                 | ✅ Done |
| Docker Containerization               | ✅ Done |
| GitHub Actions CI/CD                  | ✅ Done |
| Logging & (Optional) Monitoring       | ✅ Done |
| Summary Document                      | ✅ Included |
| 5-min Demo Video                      | ✅ Included |
| Bonus Features (Validation/Prometheus)| ✅ Done |

---

## 📝 Summary

This pipeline offers a hands-on implementation of real-world MLOps workflows. It ensures:

- Reproducibility (via DVC + MLflow)
- Deployability (via FastAPI + Docker)
- Automation (via GitHub Actions)
- Observability (via logging and metrics)

---

## 🔗 Useful Links

- [GitHub Repository](https://github.com/your-username/mlops-housing-assignment)
- [Docker Hub Image](https://hub.docker.com/r/your-docker-id/mlops-api)

---

## 📄 License
This project is submitted for educational purposes only as part of the MLOps course assignment.
