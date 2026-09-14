# Network Anomaly Detection — Production-Grade ML System

An end-to-end, production-oriented machine learning system for detecting anomalies in network telemetry data, built on the CICIDS2017 dataset. This project covers the full ML lifecycle: data pipeline, model development, experiment tracking, real-time serving, containerization, automated testing/CI-CD, monitoring, and automated retraining.

## Overview

Network systems continuously generate telemetry — latency, packet counts, byte volumes, flags — and abnormal patterns in this data can indicate attacks, congestion, or failures. This project builds a complete pipeline that goes from raw network flow data to a live, monitored, self-retraining REST API that classifies traffic as benign or malicious, and identifies the specific attack type when applicable.

## Architecture

```
Raw CICIDS2017 CSVs
      │
      ▼
Data Cleaning & EDA (Phase 1)
      │
      ▼
Preprocessing: scaling, encoding, stratified splits (Phase 2)
      │
      ▼
Model Training: Random Forest / XGBoost / LightGBM (Phase 3)
      │
      ▼
Experiment Tracking: MLflow (Phase 4)
      │
      ▼
FastAPI Serving: binary → multiclass cascade (Phase 5)
      │
      ▼
Docker + Docker Compose (Phase 6)
      │
      ▼
Testing (pytest) + CI/CD (GitHub Actions) (Phase 7)
      │
      ▼
Monitoring: Prometheus + Grafana (Phase 8)
      │
      ▼
Automated Retraining: drift detection → retrain → compare → promote (Phase 9)
```

## Dataset

**CICIDS2017** (Canadian Institute for Cybersecurity) — labeled network flow data covering five days of realistic traffic, including benign activity and multiple attack categories: DoS (Hulk, GoldenEye, slowloris, Slowhttptest), DDoS, PortScan, Brute Force (FTP/SSH-Patator), Web Attacks (Brute Force, XSS, SQL Injection), Bot, Infiltration, and Heartbleed.

After cleaning (deduplication, inf/NaN handling, label normalization), the working dataset contains **~2.52M rows** across **78 flow-level features**.

## Modeling Approach

Two complementary classification targets:

- **Binary (primary)**: ATTACK vs. BENIGN — the core real-time detection target.
- **Multiclass (secondary)**: specific attack type, with three classes excluded (Heartbleed, Infiltration, Web Attack - SQL Injection) due to having fewer than 50 samples, insufficient for reliable stratified evaluation.

Three algorithms were trained and compared for the binary task (Random Forest, XGBoost, LightGBM), all achieving near-perfect precision/recall (~0.99–1.00). This is consistent with documented critiques of CICIDS2017's synthetic attack generation producing highly separable traffic, rather than indicating unrealistic real-world performance. The multiclass model revealed more genuine, informative weaknesses — notably confusion between Web Attack - XSS and Web Attack - Brute Force, which is explainable: both are HTTP-based attacks that can produce similar flow-level statistics despite differing entirely in payload content, which these features don't capture.

## Repository Structure

```
├── notebooks/              # EDA, preprocessing, and model training notebooks
├── src/
│   ├── main.py              # FastAPI application
│   ├── model_loader.py       # Loads models/scalers/encoders at startup
│   ├── schemas.py            # Pydantic request/response models
│   └── retrain_pipeline.py   # Automated retraining pipeline (Phase 9)
├── tests/                  # pytest suite (model loading + API tests)
├── data/processed/          # Trained models, scalers, encoders, feature schema
├── docker/prometheus/       # Prometheus scrape configuration
├── .github/workflows/       # CI/CD pipeline definitions
├── Dockerfile
├── docker-compose.yml       # API + Prometheus + Grafana stack
└── requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.12+
- Docker and Docker Compose

### Local development
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Full stack (API + monitoring)
```bash
docker compose up -d
```
This starts:
- **API** at `http://localhost:8000` (docs at `/docs`)
- **Prometheus** at `http://localhost:9090`
- **Grafana** at `http://localhost:3000` (default login: admin/admin)

### Running tests
```bash
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/model-info` | GET | Currently loaded models and feature schema |
| `/predict` | POST | Classify a network flow (raw, unscaled features) |
| `/metrics` | GET | Prometheus metrics (system + ML) |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [80.0, 1293792.0, ...]}'   # 78 raw feature values, in feature_columns.json order
```

**Example response:**
```json
{
  "is_attack": true,
  "attack_probability": 0.9999999528433124,
  "attack_type": "DDoS",
  "attack_type_confidence": 0.9999999862202931
}
```

Note: the API expects **raw, unscaled** feature values — scaling is applied internally to match training-time preprocessing exactly.

## Monitoring

The Grafana dashboard ("Network Anomaly API Monitoring") tracks two tiers of metrics:

**System health**: request rate, P95 latency, error rate, memory usage, CPU time.

**ML health**: prediction distribution by outcome/attack type, and the rolling distribution of attack-probability confidence scores — an early signal for model degradation or data drift.

## Automated Retraining

`src/retrain_pipeline.py` implements the full retraining loop:

1. **Validate** incoming data against the expected feature schema.
2. **Detect drift** by comparing feature means against the production scaler's baseline (flags features shifting more than 15%).
3. **Retrain** a candidate model on the new data.
4. **Evaluate** the current production model on the same new data, for a fair comparison.
5. **Compare** F1 scores; promote the candidate only if it outperforms the current model.
6. **Log** every run — win or lose — to MLflow for a full audit trail.
7. **Deploy** the promoted model; because `data/processed/` is a Docker volume mount, a container restart alone (no rebuild) picks up the new model.

Run manually:
```bash
python -m src.retrain_pipeline
```

In testing, this pipeline correctly detected a severe drift event (a 143.7% shift in one feature) that coincided with a genuine performance drop in the production model (F1 dropped from 0.997 to 0.739), and correctly promoted a retrained replacement that recovered performance — validating the drift-detection-to-promotion loop end to end.

## Known Limitations

- CICIDS2017's synthetic attack traffic is more separable than real-world attacks would be; near-perfect binary classification results should be read in that context, not as a claim of production-grade real-world performance.
- Flow-level features cannot distinguish attacks that differ only in payload content (e.g., Web Attack subtypes), since no packet payload information is captured.
- Classes with fewer than 50 samples were excluded from multiclass evaluation due to insufficient data for reliable stratified splitting.
- Drift detection uses a simple feature-mean-shift heuristic; a production system would likely use a more statistically rigorous method (e.g., population stability index, KS-test per feature).

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · MLflow · FastAPI · Docker · Docker Compose · pytest · GitHub Actions · Prometheus · Grafana
