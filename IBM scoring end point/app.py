from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import requests
import json

app = Flask(__name__)

# -------------------------------------------------------
# IBM Watson ML Configuration
# Replace with your actual IBM Cloud credentials
# -------------------------------------------------------
IBM_API_KEY    = "YOUR_IBM_API_KEY"
IBM_ENDPOINT   = "YOUR_IBM_SCORING_ENDPOINT"
TOKEN_URL      = "https://iam.cloud.ibm.com/identity/token"

def get_ibm_token(api_key):
    """Get IBM IAM access token."""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data    = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    resp    = requests.post(TOKEN_URL, headers=headers, data=data)
    return resp.json().get("access_token")

def predict_via_ibm(features):
    """Send features to IBM Watson ML scoring endpoint."""
    token   = get_ibm_token(IBM_API_KEY)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json"
    }
    payload = {"input_data": [{"fields": [
        "ambient","coolant","u_d","u_q","motor_speed",
        "torque","i_d","i_q","stator_yoke","stator_tooth","stator_winding"
    ], "values": [features]}]}
    resp = requests.post(IBM_ENDPOINT, headers=headers, json=payload)
    result = resp.json()
    return result["predictions"][0]["values"][0][0]

# -------------------------------------------------------
# Fallback: local model (if IBM credentials not set)
# -------------------------------------------------------
try:
    with open('../model.save', 'rb') as f:
        local_model = pickle.load(f)
    with open('../transform.save', 'rb') as f:
        local_scaler = pickle.load(f)
    USE_LOCAL = True
except:
    USE_LOCAL = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['ambient']),
            float(request.form['coolant']),
            float(request.form['u_d']),
            float(request.form['u_q']),
            float(request.form['motor_speed']),
            float(request.form['torque']),
            float(request.form['i_d']),
            float(request.form['i_q']),
            float(request.form['stator_yoke']),
            float(request.form['stator_tooth']),
            float(request.form['stator_winding']),
        ]

        if IBM_API_KEY != "YOUR_IBM_API_KEY":
            prediction = predict_via_ibm(features)
            source = "IBM Watson ML"
        elif USE_LOCAL:
            arr   = np.array([features])
            arr   = local_scaler.transform(arr)
            prediction = local_model.predict(arr)[0]
            source = "Local Model"
        else:
            return render_template('index.html', prediction_text="Error: No model available.")

        return render_template('index.html',
                               prediction_text=f'[{source}] Predicted Rotor Temperature: {prediction:.2f} °C')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for JSON requests."""
    data     = request.json
    features = data.get('features', [])
    if len(features) != 11:
        return jsonify({"error": "Expected 11 feature values"}), 400
    if USE_LOCAL:
        arr        = np.array([features])
        arr        = local_scaler.transform(arr)
        prediction = local_model.predict(arr)[0]
        return jsonify({"predicted_temperature": round(float(prediction), 2)})
    return jsonify({"error": "Model not loaded"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
