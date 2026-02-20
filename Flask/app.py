from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('model.save', 'rb') as f:
    model = pickle.load(f)

with open('transform.save', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
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

        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted Rotor Temperature: {prediction:.2f} °C')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
