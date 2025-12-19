# Drug Review Classification

An end-to-end machine learning classification system that predicts drug effectiveness from patient reviews using NLP, scikit-learn, XGBoost, FastAPI, Streamlit, Docker, and a reproducible experiment pipeline.

## 📋 Project Overview

This project uses the **Drug Reviews dataset** (DrugLib.com) to predict drug effectiveness based on patient reviews. The target variable is **effectiveness** with 5 classes:
- Ineffective
- Marginally Effective
- Moderately Effective
- Considerably Effective
- Highly Effective

## 🏗️ Project Structure

```
drug_review_classification/
├── api/
│   ├── Dockerfile
│   ├── app.py                 # FastAPI prediction service
│   ├── drug_pipeline.py       # Shared pipeline components
│   └── requirements.txt
├── data/
│   └── data_schema.json       # Feature schema for Streamlit
├── models/
│   └── *.pkl                  # Trained model files
├── notebooks/
│   ├── 01_create_database.ipynb
│   ├── 02_train_model_without_optuna.ipynb
│   ├── 03_train_models_with_optuna.ipynb
│   └── 04_generate_streamlit_options.ipynb
├── streamlit/
│   ├── Dockerfile
│   ├── app.py                 # Streamlit frontend
│   └── requirements.txt
├── docker-compose.yml
├── drug_pipeline.py           # Root-level pipeline module
├── .gitignore
└── README.md
```

## 🧪 Experiments

The project runs **16 experiments** with 4 classification models, each evaluated under 4 conditions:

### Models
1. Logistic Regression
2. Ridge Classifier
3. HistGradientBoostingClassifier
4. XGBoost

### Conditions
1. No PCA + No Optuna (baseline)
2. With PCA + No Optuna
3. No PCA + With Optuna tuning
4. With PCA + With Optuna tuning

### Metrics
- **Primary Metric**: F1-score (macro)
- All 16 F1 scores are logged to MLflow/Dagshub

## 🗄️ Database

The project uses a **normalized 3NF SQLite database** as the data source:

### Tables
- `drugs` - Drug dimension table
- `conditions` - Medical condition dimension table
- `side_effects` - Side effects severity dimension table
- `effectiveness_levels` - Effectiveness level dimension table
- `reviews` - Fact table with reviews and foreign keys

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/teja4848/drug_reviw_classification.git
cd drug_reviw_classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r api/requirements.txt
```

### Running Notebooks

Open the notebooks in Google Colab or Jupyter:

1. `01_create_database.ipynb` - Creates the SQLite database
2. `02_train_model_without_optuna.ipynb` - Trains 8 models without hyperparameter tuning
3. `03_train_models_with_optuna.ipynb` - Trains 8 models with Optuna hyperparameter tuning
4. `04_generate_streamlit_options.ipynb` - Generates the data schema for Streamlit

### Running with Docker

Build and run the services:

```bash
docker-compose up --build
```

Access the applications:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs (if exposed)

### Running Locally

1. Start the FastAPI server:
```bash
cd api
uvicorn app:app --reload --port 8000
```

2. Start the Streamlit app:
```bash
cd streamlit
streamlit run app.py
```

## 📊 MLflow / Dagshub Tracking

Experiments are tracked using MLflow with Dagshub as the backend.

### Setup
1. Create a `.env` file with your Dagshub credentials:
```
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
```

2. Each experiment logs:
- `model_family`: Model type (logistic, ridge, etc.)
- `uses_pca`: Whether PCA was used (true/false)
- `is_tuned`: Whether Optuna was used (true/false)
- `cv_f1`: Cross-validation F1 score
- `test_f1`: Test set F1 score

## 🐳 Deployment

The project is containerized with Docker:

### Services
- **api**: FastAPI backend for predictions
- **streamlit**: Streamlit frontend for user interaction

### Deployment Options
- DigitalOcean
- AWS ECS
- Google Cloud Run
- Any Docker-compatible platform

## 📝 API Endpoints

### POST /predict
Predict drug effectiveness from review data.

**Request:**
```json
{
  "instances": [
    {
      "urlDrugName": "Lipitor",
      "condition": "High Cholesterol",
      "benefitsReview": "Lowered my cholesterol significantly.",
      "sideEffectsReview": "Minor muscle aches.",
      "commentsReview": "Overall satisfied.",
      "rating": 8.0,
      "sideEffects": "Mild Side Effects"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": ["Considerably Effective"],
  "probabilities": [
    {
      "Ineffective": 0.02,
      "Marginally Effective": 0.08,
      "Moderately Effective": 0.15,
      "Considerably Effective": 0.55,
      "Highly Effective": 0.20
    }
  ]
}
```

### GET /health
Health check endpoint for container orchestration.

### GET /model-info
Get information about the loaded model.

## 📚 Dataset

The Drug Reviews dataset is from DrugLib.com and contains:
- Drug names
- Medical conditions
- Patient reviews (benefits, side effects, comments)
- Ratings (1-10)
- Effectiveness labels (target variable)

## 🛠️ Technologies

- **ML/Data**: scikit-learn, XGBoost, LightGBM, pandas, numpy
- **NLP**: TF-IDF vectorization
- **Experiment Tracking**: MLflow, Dagshub
- **Hyperparameter Tuning**: Optuna
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite
- **Containerization**: Docker, Docker Compose

## 📄 License

This project is for educational purposes as part of a machine learning course.

## 👤 Author

Teja - [GitHub](https://github.com/teja4848)
