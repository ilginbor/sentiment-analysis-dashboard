import re
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import text

from app.database import get_engine


MODEL_FILE = Path("models/best_model.joblib")


def clean_text(text_value: str) -> str:
    text_value = text_value.lower()
    text_value = re.sub(r"http\S+|www\S+", " ", text_value)
    text_value = re.sub(r"[^a-zA-Z\s]", " ", text_value)
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Best model file not found. Run: python scripts\\train_models.py"
        )

    model_package = joblib.load(MODEL_FILE)
    return model_package


def get_prediction_confidence(pipeline, cleaned_text: str):
    """
    Logistic Regression ve Naive Bayes predict_proba destekler.
    LinearSVC desteklemez. Bu durumda karar skorundan yaklaşık güven hesaplanır.
    """

    model = pipeline.named_steps["model"]

    if hasattr(model, "predict_proba"):
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        confidence = float(np.max(probabilities))
    else:
        decision_scores = pipeline.decision_function([cleaned_text])

        if len(decision_scores.shape) == 1:
            score = abs(float(decision_scores[0]))
            confidence = 1 / (1 + np.exp(-score))
        else:
            score = float(np.max(np.abs(decision_scores)))
            confidence = 1 / (1 + np.exp(-score))

    return confidence


def classify_review(review_text: str):
    model_package = load_model()

    pipeline = model_package["pipeline"]
    model_name = model_package["model_name"]
    neutral_threshold = model_package.get("neutral_threshold", 0.60)

    cleaned_text = clean_text(review_text)

    if len(cleaned_text) < 3:
        return {
            "review_text": review_text,
            "cleaned_text": cleaned_text,
            "predicted_sentiment": "Neutral",
            "confidence_score": 0.0,
            "model_name": model_name,
        }

    raw_prediction = pipeline.predict([cleaned_text])[0]
    confidence_score = get_prediction_confidence(pipeline, cleaned_text)

    if confidence_score < neutral_threshold:
        predicted_sentiment = "Neutral"
    else:
        predicted_sentiment = raw_prediction

    return {
        "review_text": review_text,
        "cleaned_text": cleaned_text,
        "predicted_sentiment": predicted_sentiment,
        "confidence_score": confidence_score,
        "model_name": model_name,
    }


def save_prediction_to_database(prediction_result):
    engine = get_engine()

    query = text(
        """
        INSERT INTO reviews
        (review_text, predicted_sentiment, confidence_score)
        VALUES
        (:review_text, :predicted_sentiment, :confidence_score)
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "review_text": prediction_result["review_text"],
                "predicted_sentiment": prediction_result["predicted_sentiment"],
                "confidence_score": float(prediction_result["confidence_score"]),
            },
        )


def classify_and_save(review_text: str):
    prediction_result = classify_review(review_text)
    save_prediction_to_database(prediction_result)
    return prediction_result


if __name__ == "__main__":
    sample_review = "This product is amazing. I really love it."
    result = classify_and_save(sample_review)

    print("Prediction completed.")
    print(f"Review: {result['review_text']}")
    print(f"Sentiment: {result['predicted_sentiment']}")
    print(f"Confidence: {result['confidence_score']:.4f}")
    print(f"Model: {result['model_name']}")