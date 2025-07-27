# scripts/predict.py

import pickle
from utils.text_utils import preprocess_text

with open("models/tfidf_logistic.pkl", "rb") as f:
    model = pickle.load(f)

def predict_humor(text):
    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    return "Humorous 😄" if prediction == 1 else "Not Humorous 😐"

if __name__ == "__main__":
    sample = input("Enter a sentence: ")
    print("Prediction:", predict_humor(sample))
