# MLOPS Pipeline — Spam Detection

An end-to-end MLOps pipeline for SMS spam detection using DVC for pipeline orchestration, experiment tracking, and data versioning.

---

## Project Overview

This project builds a complete machine learning pipeline that classifies SMS messages as spam or ham. It demonstrates MLOps best practices including reproducible pipelines with DVC, experiment tracking with DVCLive, and structured logging across all stages.

---

## Project Structure

```
MLOPS-Pipeline/
├── src/
│   ├── data_ingestion.py       # Fetch, clean and split raw data
│   ├── data_preprocessing.py   # Text normalization and encoding
│   ├── feature_engineering.py  # TF-IDF vectorization
│   ├── model_building.py       # Train RandomForest classifier
│   └── model_evaluation.py     # Evaluate and log metrics
├── data/
│   ├── raw/                    # train.csv, test.csv (DVC tracked)
│   ├── interim/                # Preprocessed data (DVC tracked)
│   └── processed/              # TF-IDF features (DVC tracked)
├── models/
│   └── model.pkl               # Trained model (DVC tracked)
├── reports/
│   └── metrics.json            # Evaluation metrics
├── logs/                       # Per-stage log files
├── params.yaml                 # Pipeline hyperparameters
├── dvc.yaml                    # DVC pipeline definition
└── dvclive/                    # DVCLive experiment tracking output
```

---

## Pipeline Stages

### 1. Data Ingestion (`src/data_ingestion.py`)
- Fetches the SMS Spam Collection dataset from GitHub (raw CSV)
- Drops unused columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renames columns to `target` and `text`
- Splits into train/test sets using `test_size` from `params.yaml`
- Saves to `data/raw/train.csv` and `data/raw/test.csv`

### 2. Data Preprocessing (`src/data_preprocessing.py`)
- Encodes the target column using `LabelEncoder` (ham=0, spam=1)
- Removes duplicate rows
- Applies text transformation pipeline to each message:
  - Lowercasing
  - Tokenization (NLTK)
  - Removes non-alphanumeric tokens
  - Removes English stopwords and punctuation
  - Porter Stemming
- Saves to `data/interim/train_processed.csv` and `test_processed.csv`

### 3. Feature Engineering (`src/feature_engineering.py`)
- Applies TF-IDF vectorization using `TfidfVectorizer`
- `max_features` is configurable via `params.yaml`
- Fits on training data, transforms both train and test
- Saves feature matrices to `data/processed/train_tfidf.csv` and `test_tfidf.csv`

### 4. Model Building (`src/model_building.py`)
- Loads TF-IDF processed training data
- Trains a `RandomForestClassifier` with configurable hyperparameters
- Saves the trained model to `models/model.pkl` using pickle

### 5. Model Evaluation (`src/model_evaluation.py`)
- Loads the trained model and test features
- Computes evaluation metrics: accuracy, precision, recall, AUC
- Logs metrics and params using DVCLive for experiment tracking
- Saves metrics to `reports/metrics.json`

---

## Parameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.10         # Fraction of data used for testing

feature_engineering:
  max_features: 30        # Max TF-IDF vocabulary size

model_building:
  n_estimators: 20        # Number of trees in RandomForest
  random_state: 2         # Reproducibility seed
```

---

## Setup

### Prerequisites
- Python 3.8+
- DVC

### Install dependencies

```bash
pip install pandas scikit-learn nltk pyyaml dvc dvclive
```

### Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Running the Pipeline

Run the full pipeline using DVC:

```bash
dvc exp run
```

Run a specific stage:

```bash
dvc repro <stage_name>
# e.g.
dvc repro data_ingestion
```

Run an experiment with a parameter override:

```bash
dvc exp run --set-param model_building.n_estimators=100
```

Compare experiments:

```bash
dvc exp show
```

---

## Logging

Each stage writes structured logs to the `logs/` directory:

| Stage | Log File |
|---|---|
| Data Ingestion | `logs/data_ingestion.log` |
| Data Preprocessing | `logs/data_preprocessing.log` |
| Feature Engineering | `logs/feature_engineering.log` |
| Model Building | `logs/model_building.log` |
| Model Evaluation | `logs/model_evaluation.log` |

---

## Dataset

- Source: [SMS Spam Collection Dataset](https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv)
- ~5,500 SMS messages labeled as `ham` or `spam`
This project covers an end-to-end understanding of building a machine learning pipeline and experimenting with DVC experiments and data versioning using AWS S3.


