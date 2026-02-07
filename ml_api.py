from flask import Flask, request, jsonify
import joblib
from feature_extractor import extract_features

app = Flask(__name__)
model = joblib.load("scam_detector_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = extract_features(
        data.get("content",""),
        data.get("manual_score", 0)
    )
    prob = model.predict_proba([features])[0][1]
    return jsonify({
        "probability": prob,
        "risk_score": round(prob * 100)
    })

if __name__ == "__main__":
    app.run(port=5000)
