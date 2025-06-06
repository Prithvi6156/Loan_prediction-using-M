from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and preprocessing steps
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
pca = joblib.load("pca.pkl")

# Serve the frontend
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Backend validation: Ensure all fields are filled
        for key, value in data.items():
            if value == "" or value is None:
                return jsonify({"error": f"Missing value for {key.replace('_', ' ').capitalize()}"}), 400

        # Convert inputs to numeric
        features = np.array([
            float(data["dependents"]), float(data["education"]), float(data["self_employed"]),
            float(data["income"]), float(data["loan_amount"]), float(data["loan_term"]),
            float(data["cibil_score"]), float(data["res_assets"]), float(data["com_assets"]),
            float(data["lux_assets"]), float(data["bank_assets"])
        ]).reshape(1, -1)

        # Apply preprocessing
        features = scaler.transform(features)
        features = selector.transform(features)
        features = pca.transform(features)

        # Get probability of approval
        approval_prob = model.predict_proba(features)[0][1]  # Probability of approval
        probability_percentage = round(approval_prob * 100, 2)

        # Determine result
        if approval_prob > 0.5:
            result = f"✅ Approved (Confidence: {probability_percentage}%)"
        else:
            result = f"❌ Rejected (Confidence: {probability_percentage}%)"

        return jsonify({"prediction": result, "approval_probability": probability_percentage})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
