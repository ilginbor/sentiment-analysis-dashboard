from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.database import get_engine


DATA_FILE = Path("data/processed/prepared_reviews.csv")
MODELS_DIR = Path("models")
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
RESULTS_FILE = MODELS_DIR / "model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. Run: python scripts\\prepare_data.py"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = ["tokenized_text", "sentiment"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in prepared dataset: {missing_columns}. "
            "Run python scripts\\prepare_data.py again."
        )

    df = df.dropna(subset=["tokenized_text", "sentiment"])

    print(f"Loaded rows: {len(df)}")
    print(df["sentiment"].value_counts())

    return df


def ensure_confusion_matrix_table():
    engine = get_engine()

    query = text(
        """
        IF OBJECT_ID('model_confusion_matrices', 'U') IS NULL
        BEGIN
            CREATE TABLE model_confusion_matrices (
                matrix_id INT IDENTITY(1,1) PRIMARY KEY,
                model_name NVARCHAR(100) NOT NULL,
                actual_sentiment NVARCHAR(20) NOT NULL,
                predicted_sentiment NVARCHAR(20) NOT NULL,
                count_value INT NOT NULL,
                created_at DATETIME2 DEFAULT SYSDATETIME()
            );
        END
        """
    )

    with engine.begin() as connection:
        connection.execute(query)

    print("model_confusion_matrices table is ready.")


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": LinearSVC(),
    }

    pipelines = {}

    for model_name, model in models.items():
        pipelines[model_name] = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20000,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", model),
            ]
        )

    return pipelines


def save_result_to_database(model_name, accuracy, precision, recall, f1):
    engine = get_engine()

    insert_query = text(
        """
        INSERT INTO model_results
        (model_name, accuracy, precision_score, recall_score, f1_score)
        VALUES
        (:model_name, :accuracy, :precision_score, :recall_score, :f1_score)
        """
    )

    with engine.begin() as connection:
        connection.execute(
            insert_query,
            {
                "model_name": model_name,
                "accuracy": float(accuracy),
                "precision_score": float(precision),
                "recall_score": float(recall),
                "f1_score": float(f1),
            },
        )


def save_confusion_matrix_to_database(model_name, labels, matrix):
    engine = get_engine()

    delete_query = text(
        """
        DELETE FROM model_confusion_matrices
        WHERE model_name = :model_name
        """
    )

    insert_query = text(
        """
        INSERT INTO model_confusion_matrices
        (model_name, actual_sentiment, predicted_sentiment, count_value)
        VALUES
        (:model_name, :actual_sentiment, :predicted_sentiment, :count_value)
        """
    )

    with engine.begin() as connection:
        connection.execute(delete_query, {"model_name": model_name})

        for actual_index, actual_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                connection.execute(
                    insert_query,
                    {
                        "model_name": model_name,
                        "actual_sentiment": actual_label,
                        "predicted_sentiment": predicted_label,
                        "count_value": int(matrix[actual_index][predicted_index]),
                    },
                )


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_confusion_matrix_table()

    df = load_data()

    X = df["tokenized_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_models()

    results = []
    best_model_name = None
    best_model = None
    best_f1 = -1

    labels = sorted(y_test.unique())

    for model_name, pipeline in pipelines.items():
        print("=" * 60)
        print(f"Training model: {model_name}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        matrix = confusion_matrix(y_test, y_pred, labels=labels)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(matrix)

        save_result_to_database(model_name, accuracy, precision, recall, f1)
        save_confusion_matrix_to_database(model_name, labels, matrix)

        results.append(
            {
                "model_name": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = pipeline

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_model,
            "neutral_threshold": 0.60,
        },
        BEST_MODEL_FILE,
    )

    print("=" * 60)
    print("Training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Saved best model to: {BEST_MODEL_FILE}")
    print(f"Saved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    train_and_evaluate()