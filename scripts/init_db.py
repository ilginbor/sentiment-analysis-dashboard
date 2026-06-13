import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.database import get_engine


PREPARED_DATA_FILE = Path("data/processed/prepared_reviews.csv")


def create_prepared_reviews_table():
    engine = get_engine()

    query = text(
        """
        IF OBJECT_ID('prepared_reviews', 'U') IS NOT NULL
        BEGIN
            DROP TABLE prepared_reviews;
        END;

        CREATE TABLE prepared_reviews (
            prepared_review_id INT IDENTITY(1,1) PRIMARY KEY,
            review_text NVARCHAR(MAX) NOT NULL,
            cleaned_text NVARCHAR(MAX) NOT NULL,
            tokenized_text NVARCHAR(MAX) NOT NULL,
            sentiment NVARCHAR(20) NOT NULL,
            created_at DATETIME2 DEFAULT SYSDATETIME()
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(query)

    print("prepared_reviews table is ready.")


def load_prepared_reviews_to_database():
    if not PREPARED_DATA_FILE.exists():
        raise FileNotFoundError(
            "Prepared CSV file not found. Run: python scripts\\prepare_data.py"
        )

    engine = get_engine()

    df = pd.read_csv(PREPARED_DATA_FILE)

    required_columns = [
        "review_text",
        "cleaned_text",
        "tokenized_text",
        "sentiment",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in prepared CSV: {missing_columns}. "
            "Run python scripts\\prepare_data.py again."
        )

    df = df[required_columns]
    df = df.dropna()

    print(f"Loading {len(df)} rows into prepared_reviews table...")

    df.to_sql(
        name="prepared_reviews",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
    )

    print("Prepared reviews loaded successfully.")


def main():
    create_prepared_reviews_table()
    load_prepared_reviews_to_database()


if __name__ == "__main__":
    main()