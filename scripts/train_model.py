# scripts/train_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scripts.preprocess import load_and_preprocess_data


data_path = "data/processed/processed_data(20k).csv"
df = load_and_preprocess_data(data_path)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

with open("models/tfidf_logistic.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved to models/tfidf_logistic.pkl")
