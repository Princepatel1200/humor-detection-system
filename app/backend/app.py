# app/backend/app.py
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flask import Flask, request, render_template
import pickle
from utils.text_utils import preprocess_text


MODEL_PATH = os.path.join("models", "tfidf_logistic.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder="../frontend")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    label = "Humorous 😄" if prediction == 1 else "Not Humorous 😐"
    return render_template("index.html", input_text=text, prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
