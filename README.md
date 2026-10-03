# Beverage Price Prediction

This project builds an end-to-end machine learning workflow for predicting a customer's preferred beverage price range from survey responses. It covers data cleaning, feature engineering, model comparison, experiment tracking, and model versioning.

MLflow records every model's parameters, evaluation metrics, classification report, confusion matrix, and serialized pipeline. DagsHub hosts the public experiment interface and registered model versions.

## Workflow

1. Remove duplicate and logically invalid records.
2. Impute missing categorical values and standardize misspelled categories.
3. Engineer age-group, consume-frequency/brand-awareness, zone-affluence, and brand-switching features.
4. Split the cleaned data into training and test sets.
5. Train and compare six classification algorithms.
6. Track metrics and artifacts with MLflow and register the models on DagsHub.

## Models

- Gaussian Naive Bayes
- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost
- LightGBM

## Results

| Model | Test accuracy |
| --- | ---: |
| XGBoost | 92.26% |
| LightGBM | 92.16% |
| Random Forest | 89.02% |
| Support Vector Machine | 82.45% |
| Logistic Regression | 80.06% |
| Gaussian Naive Bayes | 58.31% |

XGBoost achieved the highest test accuracy. All six runs and registered model versions can be reviewed in the [public MLflow experiment](https://dagshub.com/sarthak-here/beverage-price-prediction-mlflow.mlflow/#/experiments/0/runs).

## Project structure

```text
beverage-price-prediction/
|-- train_mlflow.py
|-- app.py
|-- model/
|   `-- xgboost_pipeline.pkl
|-- requirements.txt
|-- data/
|   `-- README.md
|-- .gitignore
`-- README.md
```

## Interactive application

The Streamlit application uses the tracked XGBoost pipeline to generate live price-range predictions and display the full class-probability distribution.

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Client presentation

The `presentation/` folder contains:

- An editable seven-slide PowerPoint focused on the business problem, recommendation, and rollout plan.

## Data privacy

The source survey is not committed to GitHub or logged as an MLflow artifact. Runs contain aggregate metrics, evaluation reports, plots, and fitted model pipelines only.

## Setup

```powershell
C:\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy the source data to `data/survey_results.csv`.

## Local validation

```powershell
.\.venv\Scripts\python.exe train_mlflow.py --tracking-uri sqlite:///mlflow.db
```

Launch the local interface with:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

## DagsHub tracking

Authenticate with the DagsHub client, then run:

```powershell
.\.venv\Scripts\python.exe train_mlflow.py `
  --tracking-uri https://dagshub.com/sarthak-here/beverage-price-prediction-mlflow.mlflow `
  --dagshub-owner sarthak-here `
  --dagshub-repo beverage-price-prediction-mlflow `
  --register-models
```

The script uses the experiment name `Beverage Price Prediction` and logs one run per model.
