import bz2
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DATA_DIR / "prepared_reviews.csv"

MAX_ROWS = 50000


def clean_text(text: str) -> str:
    """
    Review metnini temizler:
    - küçük harfe çevirir
    - URL'leri siler
    - özel karakterleri ve sayıları temizler
    - fazla boşlukları kaldırır
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text: str) -> list[str]:
    custom_stop_words = {
        "the", "and", "is", "it", "i", "a", "an", "to", "of", "for", "in",
        "on", "this", "that", "with", "was", "were", "be", "have", "has",
        "had", "my", "you", "your", "but", "very", "after", "one", "day",
        "ever", "really", "just", "also", "than", "then", "there", "their",
        "they", "them", "are", "am", "as", "at", "by", "from", "or", "so",
        "if", "not", "can", "will", "would", "could", "should"
    }

    stop_words = set(ENGLISH_STOP_WORDS).union(custom_stop_words)

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return tokens

def tokens_to_text(tokens: list[str]) -> str:
    """
    Token listesini tekrar boşluklarla ayrılmış metin haline getirir.
    Bu alan TF-IDF model eğitiminde kullanılacaktır.
    """
    return " ".join(tokens)


def parse_amazon_review_line(line: str):
    """
    Amazon Reviews dataset format:
    __label__1 negative review text
    __label__2 positive review text
    """
    line = line.strip()

    if line.startswith("__label__1"):
        label = "Negative"
        text = line.replace("__label__1", "", 1).strip()
    elif line.startswith("__label__2"):
        label = "Positive"
        text = line.replace("__label__2", "", 1).strip()
    else:
        return None, None

    return text, label


def find_dataset_file() -> Path:
    possible_files = [
        RAW_DATA_DIR / "train.ft.txt.bz2",
        RAW_DATA_DIR / "test.ft.txt.bz2",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        "Dataset file not found. Expected train.ft.txt.bz2 or test.ft.txt.bz2 inside data/raw."
    )


def prepare_dataset():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_file = find_dataset_file()
    print(f"Reading dataset from: {dataset_file}")

    rows = []

    with bz2.open(dataset_file, "rt", encoding="utf-8", errors="ignore") as file:
        for index, line in enumerate(file):
            if index >= MAX_ROWS:
                break

            review_text, sentiment = parse_amazon_review_line(line)

            if review_text is None:
                continue

            cleaned_text = clean_text(review_text)

            if len(cleaned_text) < 5:
                continue

            tokens = tokenize_text(cleaned_text)
            tokenized_text = tokens_to_text(tokens)

            if len(tokens) == 0:
                continue

            rows.append(
                {
                    "review_text": review_text,
                    "cleaned_text": cleaned_text,
                    "tokenized_text": tokenized_text,
                    "sentiment": sentiment,
                }
            )

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["tokenized_text"])
    df = df.dropna()

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Dataset prepared successfully.")
    print(f"Rows: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("Columns:", list(df.columns))
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    prepare_dataset()