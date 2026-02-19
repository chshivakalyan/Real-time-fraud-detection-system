#  Real-Time Credit Card Fraud Detection System

> End-to-end Machine Learning + MLOps pipeline for real-time fraud detection using the IEEE-CIS Fraud Detection dataset.

---

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Decision Policy](#decision-policy)
- [Monitoring](#monitoring)
- [Docker](#docker)
- [AWS Deployment](#aws-deployment)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)
- [Author](#author)

---

## Overview

This project implements a **production-style fraud detection system** covering the full ML lifecycle — from data preprocessing and model training to containerized deployment and live monitoring.

**Core capabilities:**

- Data preprocessing and feature engineering on a highly imbalanced dataset
- XGBoost model training with class imbalance handling
- Model versioning with artifact management
- FastAPI real-time inference API
- Streamlit frontend for live scoring
- Monitoring pipeline: prediction logging, drift detection, performance checks
- Docker containerization and AWS EC2 deployment

---

## Architecture

```
User → Streamlit UI → FastAPI (EC2) → XGBoost Model → Prediction → Logs
                                                               ↓
                                                     Monitoring Scripts
                                               (Drift Detection · Performance)
```

Requests flow from the Streamlit frontend to a containerized FastAPI service running on AWS EC2. Each prediction is logged for downstream monitoring and alerting.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | IEEE-CIS Fraud Detection Dataset |
| Task | Binary classification (`isFraud`) |
| Challenge | Highly imbalanced classes |

---

## Project Structure

```
ML/
│
├── api/
│   └── app.py                  # FastAPI application
│
├── src/
│   ├── train.py                # Model training pipeline
│   ├── inference.py            # Inference logic
│   └── preprocessing.py        # Feature engineering & preprocessing
│
├── monitoring/
│   ├── drift_check.py          # KS-test based drift detection
│   └── performance_check.py    # Model performance evaluation
│
├── models/                     # Saved model artifacts
├── logs/                       # Prediction logs
│
├── streamlit_app.py            # Frontend UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Model artifacts saved to `models/`:**

- `xgb_fraud_model.pkl` — trained XGBoost model
- `preprocessor.pkl` — fitted preprocessing pipeline
- `feature_columns.json` — ordered feature schema
- `model_metadata.json` — version and training metadata

---

## Model Details

| Property | Detail |
|----------|--------|
| Algorithm | XGBoost (`binary:logistic`) |
| Imbalance handling | `scale_pos_weight` |
| Optimization | CPU-optimized (`tree_method="hist"`) |

---

## Getting Started

### 1. Train the Model

```bash
python src/train.py
```

This will train the XGBoost model, save all artifacts to `models/`, and update version metadata.

### 2. Run the API Locally

```bash
uvicorn api.app:app --reload
```

Swagger UI available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Launch the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

The UI connects to the deployed FastAPI endpoint for live fraud scoring.

---

## API Reference

### `POST /predict`

**Example Request:**

```json
{
  "TransactionAmt": 15000,
  "ProductCD": "W",
  "card1": 9999,
  "card4": "visa",
  "card6": "credit",
  "DeviceType": "desktop",
  "P_emaildomain": "mailinator.com",
  "R_emaildomain": "mailinator.com"
}
```

**Example Response:**

```json
{
  "fraud_probability": 0.82,
  "decision": "block",
  "model_version": "v2"
}
```

---

## Decision Policy

| Fraud Probability | Decision |
|:-----------------:|:--------:|
| < 0.3 |Allow |
| 0.3 – 0.6 | Challenge |
| > 0.6 |Block |

---

## Monitoring

### Prediction Logging

Every API request is logged to `logs/predictions.csv` with:

- Timestamp
- Fraud probability
- Decision (allow / challenge / block)
- Model version
- Input features

### Drift Detection

Uses the **Kolmogorov–Smirnov test** to detect numeric feature drift between training distribution and live traffic.

```bash
python monitoring/drift_check.py
```

### Performance Monitoring

Evaluates model performance when ground truth labels become available.

```bash
python monitoring/performance_check.py
```

---

## Docker

**Build the image:**

```bash
docker build -t fraud-api .
```

**Run the container:**

```bash
docker run -p 80:8000 fraud-api
```

---

## AWS Deployment

| Property | Detail |
|----------|--------|
| Platform | AWS EC2 (Amazon Linux) |
| Service | Dockerized FastAPI |
| Port | 80 (publicly exposed) |
| Access | EC2 public IP |
| Observed latency | ~700 ms (network + EC2 overhead) |

> **Why not SageMaker?** Manual Docker + EC2 deployment was intentionally chosen to demonstrate infrastructure control, containerization skills, deployment mechanics, and custom monitoring — without relying on managed ML services.

---

## Future Improvements

- [ ] Automated retraining pipeline
- [ ] MLflow experiment tracking
- [ ] CI/CD with GitHub Actions
- [ ] Load balancing
- [ ] Redis caching
- [ ] CloudWatch metrics integration

---

## Skills Demonstrated

`Machine Learning` · `Feature Engineering` · `Imbalanced Classification` · `FastAPI` · `Docker` · `AWS EC2` · `Model Versioning` · `Drift Detection` · `Streamlit`

---

## Author

**Shiva Kalyan**
*End-to-End Machine Learning Engineering Project*
