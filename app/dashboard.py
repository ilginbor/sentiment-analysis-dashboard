import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.database import get_engine
from app.predict import classify_and_save


st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_reviews():
    engine = get_engine()

    query = text(
        """
        SELECT
            review_id,
            review_text,
            predicted_sentiment,
            confidence_score,
            classification_time
        FROM reviews
        ORDER BY classification_time DESC
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


def load_model_results():
    engine = get_engine()

    query = text(
        """
        SELECT
            model_name,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            created_at
        FROM model_results
        ORDER BY f1_score DESC
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


def load_confusion_matrix_results():
    engine = get_engine()

    query = text(
        """
        IF OBJECT_ID('model_confusion_matrices', 'U') IS NOT NULL
            SELECT
                model_name,
                actual_sentiment,
                predicted_sentiment,
                count_value
            FROM model_confusion_matrices;
        ELSE
            SELECT
                CAST(NULL AS NVARCHAR(100)) AS model_name,
                CAST(NULL AS NVARCHAR(20)) AS actual_sentiment,
                CAST(NULL AS NVARCHAR(20)) AS predicted_sentiment,
                CAST(NULL AS INT) AS count_value
            WHERE 1 = 0;
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


def load_prepared_reviews_count():
    engine = get_engine()

    query = text(
        """
        IF OBJECT_ID('prepared_reviews', 'U') IS NOT NULL
            SELECT COUNT(*) AS row_count FROM prepared_reviews;
        ELSE
            SELECT 0 AS row_count;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).fetchone()

    return int(result[0])


def show_prediction_section():
    st.header("Yeni Yorum Analizi")

    review_text = st.text_area(
        "Analiz etmek istediğiniz müşteri yorumunu yazın:",
        height=140,
        placeholder="Example: This product is amazing. I really love it.",
    )

    analyze_button = st.button("Duygu Analizi Yap", type="primary")

    if analyze_button:
        if not review_text.strip():
            st.warning("Lütfen bir yorum girin.")
            return

        result = classify_and_save(review_text)

        sentiment = result["predicted_sentiment"]
        confidence = result["confidence_score"]

        if sentiment == "Positive":
            st.success(f"Tahmin: Positive | Güven Skoru: {confidence:.2%}")
        elif sentiment == "Negative":
            st.error(f"Tahmin: Negative | Güven Skoru: {confidence:.2%}")
        else:
            st.info(f"Tahmin: Neutral | Güven Skoru: {confidence:.2%}")

        st.caption(f"Kullanılan model: {result['model_name']}")

    st.divider()

    st.subheader("Örnek Yorumlar")

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        st.info("Positive örnek")
        st.code("I love this product, it is perfect.")

    with example_col2:
        st.error("Negative örnek")
        st.code("This item is terrible and stopped working.")

    with example_col3:
        st.warning("Neutral örnek")
        st.code("The product is okay, not amazing but usable.")


def filter_reviews(reviews_df):
    st.subheader("Filtreler")

    reviews_df["classification_time"] = pd.to_datetime(
        reviews_df["classification_time"]
    )

    sentiment_options = ["All"] + sorted(
        reviews_df["predicted_sentiment"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_sentiment = st.selectbox(
            "Duygu filtresi:",
            sentiment_options,
        )

    with col2:
        min_confidence = st.slider(
            "Minimum güven skoru:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

    filtered_df = reviews_df.copy()

    if selected_sentiment != "All":
        filtered_df = filtered_df[
            filtered_df["predicted_sentiment"] == selected_sentiment
        ]

    filtered_df = filtered_df[
        filtered_df["confidence_score"] >= min_confidence
    ]

    return filtered_df


def show_word_frequency(reviews_df):
    st.subheader("En Sık Kullanılan Kelimeler")

    all_text = " ".join(reviews_df["review_text"].dropna().astype(str)).lower()
    all_text = re.sub(r"[^a-zA-Z\s]", " ", all_text)

    stop_words = {
        "the", "and", "is", "it", "i", "a", "an", "to", "of", "for", "in",
        "on", "this", "that", "with", "was", "were", "be", "have", "has",
        "had", "my", "you", "your", "but", "very", "after", "one", "day",
        "product", "really", "ever"
    }

    words = [
        word
        for word in all_text.split()
        if len(word) > 2 and word not in stop_words
    ]

    word_counts = Counter(words).most_common(15)

    if not word_counts:
        st.info("Kelime frekansı için yeterli yorum bulunamadı.")
        return

    word_df = pd.DataFrame(word_counts, columns=["word", "count"])

    fig_words = px.bar(
        word_df,
        x="count",
        y="word",
        orientation="h",
        title="En Sık Kullanılan 15 Kelime",
    )
    fig_words.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=600,
    )

    st.plotly_chart(fig_words, width="stretch")


def show_review_analytics():
    st.header("Yorum Analitiği")

    reviews_df = load_reviews()

    if reviews_df.empty:
        st.info("Henüz analiz edilmiş yorum yok.")
        return

    filtered_df = filter_reviews(reviews_df)

    if filtered_df.empty:
        st.warning("Seçilen filtrelere uygun yorum bulunamadı.")
        return

    total_reviews = len(filtered_df)
    positive_count = int((filtered_df["predicted_sentiment"] == "Positive").sum())
    negative_count = int((filtered_df["predicted_sentiment"] == "Negative").sum())
    neutral_count = int((filtered_df["predicted_sentiment"] == "Neutral").sum())
    avg_confidence = float(filtered_df["confidence_score"].mean())

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Toplam Yorum", total_reviews)
    col2.metric("Pozitif", positive_count)
    col3.metric("Negatif", negative_count)
    col4.metric("Neutral", neutral_count)
    col5.metric("Ortalama Güven", f"{avg_confidence:.2%}")

    sentiment_counts = filtered_df["predicted_sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_pie = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Duygu Dağılımı",
        hole=0.35,
    )
    st.plotly_chart(fig_pie, width="stretch")

    daily_counts = (
        filtered_df
        .groupby([filtered_df["classification_time"].dt.date, "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )
    daily_counts.columns = ["Date", "Sentiment", "Count"]

    fig_line = px.line(
        daily_counts,
        x="Date",
        y="Count",
        color="Sentiment",
        markers=True,
        title="Günlük Duygu Eğilimi",
    )
    st.plotly_chart(fig_line, width="stretch")

    show_word_frequency(filtered_df)

    st.subheader("Son Analiz Edilen Yorumlar")

    display_df = filtered_df[
        [
            "review_id",
            "review_text",
            "predicted_sentiment",
            "confidence_score",
            "classification_time",
        ]
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Analiz Sonuçlarını CSV Olarak İndir",
        data=csv_data,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )


def show_model_results():
    st.header("Model Karşılaştırması")

    results_df = load_model_results()

    if results_df.empty:
        st.info("Henüz model sonucu bulunamadı.")
        return

    best_model = results_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("En İyi Model", best_model["model_name"])
    col2.metric("En İyi F1-Score", f"{best_model['f1_score']:.4f}")
    col3.metric("En İyi Accuracy", f"{best_model['accuracy']:.4f}")

    st.subheader("Model Performans Tablosu")

    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
    )

    metrics_df = results_df.melt(
        id_vars=["model_name"],
        value_vars=["accuracy", "precision_score", "recall_score", "f1_score"],
        var_name="Metric",
        value_name="Score",
    )

    fig_bar = px.bar(
        metrics_df,
        x="model_name",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performans Karşılaştırması",
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Confusion Matrix")

    confusion_df = load_confusion_matrix_results()

    if confusion_df.empty:
        st.info("Confusion matrix sonucu bulunamadı. Lütfen modelleri tekrar eğitin.")
        return

    model_names = sorted(confusion_df["model_name"].dropna().unique().tolist())

    selected_model = st.selectbox(
        "Confusion Matrix için model seçin:",
        model_names,
    )

    selected_matrix_df = confusion_df[
        confusion_df["model_name"] == selected_model
    ]

    matrix_pivot = selected_matrix_df.pivot(
        index="actual_sentiment",
        columns="predicted_sentiment",
        values="count_value",
    ).fillna(0)

    fig_matrix = px.imshow(
        matrix_pivot,
        text_auto=True,
        color_continuous_scale="Reds",
        aspect="auto",
        title=f"Confusion Matrix - {selected_model}",
        labels=dict(
            x="Predicted Sentiment",
            y="Actual Sentiment",
            color="Count",
        ),
    )

    fig_matrix.update_traces(
        texttemplate="%{z}",
        textfont_size=22,
    )

    fig_matrix.update_layout(
        xaxis_title="Predicted Sentiment",
        yaxis_title="Actual Sentiment",
        height=600,
    )

    st.plotly_chart(fig_matrix, width="stretch")

    st.dataframe(
        matrix_pivot,
        width="stretch",
    )


def show_dataset_summary():
    st.header("Veri Seti Özeti")

    prepared_count = load_prepared_reviews_count()
    review_count = len(load_reviews())
    model_count = len(load_model_results())

    col1, col2, col3 = st.columns(3)

    col1.metric("Hazırlanmış Eğitim Verisi", prepared_count)
    col2.metric("Dashboard Yorum Kayıtları", review_count)
    col3.metric("Eğitilen Model Sayısı", model_count)

    st.write(
        """
        Bu bölüm, projenin SQL Server veritabanında tuttuğu temel kayıt sayılarını gösterir.
        Hazırlanmış eğitim verisi `prepared_reviews` tablosunda, kullanıcı yorumları `reviews`
        tablosunda, model sonuçları ise `model_results` tablosunda saklanır.
        """
    )


def main():
    st.title("Duygu Analizi Gösterge Paneli")

    st.write(
        "Bu uygulama müşteri yorumlarını Positive, Negative veya Neutral olarak "
        "sınıflandırır ve sonuçları SQL Server veritabanında saklar."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Yorum Analizi",
            "Yorum İstatistikleri",
            "Model Sonuçları",
            "Veri Seti Özeti",
        ]
    )

    with tab1:
        show_prediction_section()

    with tab2:
        show_review_analytics()

    with tab3:
        show_model_results()

    with tab4:
        show_dataset_summary()


if __name__ == "__main__":
    main()