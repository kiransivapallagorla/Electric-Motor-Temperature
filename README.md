# Electric Motor Temperature Prediction using Machine Learning

## Project Overview
This project predicts the rotor/permanent magnet temperature of a PMSM (Permanent Magnet Synchronous Motor) using a Random Forest Regressor, deployed via a Flask web application and IBM Watson ML scoring endpoint.

---

## Project Structure
```
├── Flask/
│   ├── templates/
│   │   └── index.html          # Web UI for local Flask app
│   └── app.py                  # Flask web application
│
├── IBM scoring end point/
│   ├── templates/
│   │   └── index.html          # Web UI for IBM deployment
│   ├── app.py                  # Flask app using IBM Watson ML
│   └── IBM traing code.ipynb   # Training & IBM deployment notebook
│
├── pmsm_temperature_data.csv   # Dataset
├── Rotor Temperature Detection.ipynb  # Main ML notebook
├── model.save                  # Trained model (generated after running notebook)
└── transform.save              # Scaler (generated after running notebook)
```

---

## Setup & Installation

### Prerequisites
```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn pickle5
```

### Step 1: Train the Model
Run `Rotor Temperature Detection.ipynb` in Jupyter Notebook.  
This will generate `model.save` and `transform.save`.

### Step 2: Copy model files to Flask folder
```bash
cp model.save Flask/
cp transform.save Flask/
```

### Step 3: Run the Flask App
```bash
cd Flask
python app.py
```
Open browser at: `http://localhost:5000`

---

## Dataset
- **File:** `pmsm_temperature_data.csv`
- **Source:** Kaggle - Electric Motor Temperature Dataset
- **Features:** ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, stator_yoke, stator_tooth, stator_winding
- **Target:** `pm` (Rotor/Permanent Magnet Temperature)

---

## Model
- **Algorithm:** Random Forest Regressor
- **Preprocessing:** StandardScaler
- **Evaluation Metrics:** R², RMSE, MAE

---

## IBM Watson ML Deployment
1. Open `IBM scoring end point/IBM traing code.ipynb`
2. Add your IBM API Key and Space ID
3. Run all cells to train and deploy
4. Copy the scoring URL to `IBM scoring end point/app.py`

---

## Use Cases
- **Preventive Maintenance** – Predict overheating before failure
- **Energy Efficiency** – Keep motors at optimal temperature
- **Equipment Reliability** – Ensure safe operating ranges
