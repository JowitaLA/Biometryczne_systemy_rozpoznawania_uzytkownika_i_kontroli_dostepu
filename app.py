from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Wczytanie modelu i skalera
model = joblib.load("model.pkl")
scaler = joblib.load("skaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    budget = data['budget']
    popularity = data['popularity']
    vote_average = data['vote_average']
    vote_count = data['vote_count']

    budget_log = np.log1p(budget)
    popularity_log = np.log1p(popularity)

    features = np.array([[
        budget,
        popularity,
        vote_average,
        vote_count,
        budget_log,
        popularity_log
    ]])

    features = scaler.transform(features)
    pred = model.predict(features)

    return jsonify({
        "prediction": float(np.expm1(pred[0]))
    })

# URUCHOMIENIE NA RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)