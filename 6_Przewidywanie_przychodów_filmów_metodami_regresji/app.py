from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("skaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    features = np.array([[
        data['budget'],
        data['popularity'],
        data['vote_average'],
        data['vote_count'],
        data['budget_log'],
        data['popularity_log']
    ]])
    
    features = scaler.transform(features)
    pred = model.predict(features)
    
    return jsonify({"prediction": float(np.expm1(pred[0]))})

app.run()