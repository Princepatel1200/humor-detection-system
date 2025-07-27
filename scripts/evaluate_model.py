# scripts/evaluate_model.py

import pickle
from sklearn.metrics import accuracy_score, classification_report
from scripts.preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split

df = load_and_preprocess_data("data/processed/processed_data(20k).csv")
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)

with open("models/tfidf_logistic.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
