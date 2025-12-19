import sqlite3
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH = "data/drug_reviews.db"   # change if your db name differs
OUT_PATH = "models/tfidf_vectorizer.joblib"

def get_dataframe_from_db(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT 
            r.benefits_review as benefitsReview,
            r.side_effects_review as sideEffectsReview,
            r.comments_review as commentsReview,
            r.split
        FROM reviews r
    """, conn)
    conn.close()
    return df

df = get_dataframe_from_db(DB_PATH)

df["combined_text"] = (
    df["benefitsReview"].fillna("") + " " +
    df["sideEffectsReview"].fillna("") + " " +
    df["commentsReview"].fillna("")
)

train_text = df[df["split"] == "train"]["combined_text"].fillna("")

tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words="english"
)

tfidf.fit(train_text)

joblib.dump(tfidf, OUT_PATH)
print(f"✅ Saved TF-IDF vectorizer to: {OUT_PATH}")
print(f"Features learned: {len(tfidf.get_feature_names_out())}")
