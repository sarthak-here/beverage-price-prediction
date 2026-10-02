from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import sklearn
import xgboost
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


EXPERIMENT_NAME = "Beverage Price Prediction"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track beverage-price models with MLflow.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/survey_results.csv"),
        help="Path to the private source CSV.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="./mlruns",
        help="MLflow tracking URI. Use a DagsHub .mlflow URL for remote logging.",
    )
    parser.add_argument(
        "--register-models",
        action="store_true",
        help="Create a registered model version for each logged pipeline.",
    )
    parser.add_argument("--dagshub-owner", help="DagsHub repository owner.")
    parser.add_argument("--dagshub-repo", help="DagsHub repository name.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model names to run. By default all six models are trained.",
    )
    return parser.parse_args()


def clean_and_engineer(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path).drop_duplicates().copy()

    df["income_levels"] = df["income_levels"].fillna("Not Reported")
    for column in ["consume_frequency(weekly)", "purchase_channel"]:
        df[column] = df[column].fillna(df[column].mode()[0])

    df["zone"] = df["zone"].replace({"Metor": "Metro", "urbna": "Urban"})
    df["current_brand"] = df["current_brand"].replace(
        {"Establishd": "Established", "newcomer": "Newcomer"}
    )

    df = df[df["age"].between(18, 70)].copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56-70"],
    ).astype(str)

    frequency_score = df["consume_frequency(weekly)"].map(
        {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
    )
    awareness_score = df["awareness_of_other_brands"].map(
        {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
    )
    df["cf_ab_score"] = (
        frequency_score / (awareness_score + frequency_score)
    ).round(2)

    zone_score = df["zone"].map(
        {"Urban": 3, "Metro": 4, "Rural": 1, "Semi-Urban": 2}
    )
    income_score = df["income_levels"].map(
        {
            "<10L": 1,
            "10L - 15L": 2,
            "16L - 25L": 3,
            "26L - 35L": 4,
            "> 35L": 5,
            "Not Reported": 0,
        }
    )
    df["zas_score"] = zone_score * income_score
    df["bsi"] = (
        (df["current_brand"] != "Established")
        & df["reasons_for_choosing_brands"].isin(["Price", "Quality"])
    ).astype(int)

    logical_outlier = (df["occupation"] == "Student") & (
        df["age_group"] == "56-70"
    )
    df = df[~logical_outlier].drop(columns=["age"]).copy()

    if len(df) != 29_956:
        raise ValueError(f"Unexpected cleaned row count: {len(df)}")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    label_columns = [
        "age_group",
        "income_levels",
        "health_concerns",
        "consume_frequency(weekly)",
        "preferable_consumption_size",
    ]
    categorical_columns = list(
        X.select_dtypes(include=["object", "string", "category"]).columns
    )
    one_hot_columns = [c for c in categorical_columns if c not in label_columns]
    numeric_columns = [
        c for c in X.columns if c not in label_columns and c not in one_hot_columns
    ]

    return ColumnTransformer(
        transformers=[
            (
                "label",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                label_columns,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                one_hot_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )


def get_models() -> dict[str, object]:
    return {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2_000, random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        ),
    }


def data_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_evaluation_artifacts(
    run_dir: Path,
    model_name: str,
    y_test,
    predictions,
    class_names: list[str],
) -> None:
    report = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = run_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=35,
        ax=axis,
        colorbar=False,
    )
    axis.set_title(f"{model_name} confusion matrix")
    figure.tight_layout()
    plot_path = run_dir / "confusion_matrix.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    mlflow.log_artifact(str(plot_path), artifact_path="evaluation")


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing data file: {data_path}. See data/README.md for setup."
        )

    if bool(args.dagshub_owner) != bool(args.dagshub_repo):
        raise ValueError("Provide both --dagshub-owner and --dagshub-repo together.")
    if args.dagshub_owner and args.dagshub_repo:
        dagshub.init(
            repo_owner=args.dagshub_owner,
            repo_name=args.dagshub_repo,
            root=".",
            mlflow=True,
            dvc=False,
            patch_mlflow=False,
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = clean_and_engineer(data_path)
    X = df.drop(columns=["respondent_id", "price_range"])
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["price_range"])
    class_names = list(target_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    fingerprint = data_fingerprint(data_path)
    results: list[dict[str, object]] = []

    available_models = get_models()
    if args.models:
        unknown_models = sorted(set(args.models) - set(available_models))
        if unknown_models:
            raise ValueError(f"Unknown model names: {unknown_models}")
        selected_models = {
            name: available_models[name] for name in args.models
        }
    else:
        selected_models = available_models

    for model_name, estimator in selected_models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X)), ("model", estimator)]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision_weighted": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }
            params = {
                "model_name": model_name,
                "random_state": RANDOM_STATE,
                "test_size": 0.25,
                "cleaned_rows": len(df),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_columns": X.shape[1],
                "target_classes": len(class_names),
                "data_sha256": fingerprint,
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "lightgbm_version": lightgbm.__version__,
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "project": "beverage-price-prediction",
                    "data_privacy": "source-data-not-uploaded",
                    "task": "multiclass-classification",
                }
            )

            with tempfile.TemporaryDirectory() as temporary_directory:
                run_dir = Path(temporary_directory)
                log_evaluation_artifacts(
                    run_dir, model_name, y_test, predictions, class_names
                )

            signature = infer_signature(X_train, pipeline.predict(X_train))
            registered_name = (
                "BeveragePrice_" + "".join(character for character in model_name if character.isalnum())
                if args.register_models
                else None
            )
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                registered_model_name=registered_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            results.append({"model": model_name, **metrics})

    result_frame = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    result_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    print(result_frame.to_string(index=False))
    print(f"\nTracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Best model: {result_frame.iloc[0]['model']}")


if __name__ == "__main__":
    main()
