from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # allows your HTML frontend to call this API

# ── Load trained model and scaler ────────────────────────────────
# These are the .pkl files you downloaded from Colab
try:
    model  = pickle.load(open('heart_model.pkl',  'rb'))
    scaler = pickle.load(open('heart_scaler.pkl', 'rb'))
    print("✅ Model and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    print("   Make sure heart_model.pkl and heart_scaler.pkl are in the same folder as app.py")
    model  = None
    scaler = None

# ── Feature order must match exactly what the model was trained on ─
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def home():
    return jsonify({
        'status': 'Heart Disease Prediction API is running',
        'model_loaded': model is not None,
        'endpoints': {
            'POST /predict': 'Send patient data, get heart disease risk'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    # ── Check model is loaded ─────────────────────────────────────
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Check that pkl files exist.'
        }), 500

    # ── Get data from the request ─────────────────────────────────
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received. Send JSON.'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400

    # ── Validate all 13 features are present ──────────────────────
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({
            'error': f'Missing fields: {missing}',
            'required': FEATURE_ORDER
        }), 400

    # ── Validate all values are numbers ──────────────────────────
    try:
        values = []
        for feature in FEATURE_ORDER:
            val = float(data[feature])
            values.append(val)
    except (ValueError, TypeError) as e:
        return jsonify({
            'error': f'All values must be numbers. Problem: {str(e)}'
        }), 400

    # ── Validate ranges (basic sanity check) ─────────────────────
    ranges = {
        'age':      (1,   120),
        'sex':      (0,   1),
        'cp':       (0,   3),
        'trestbps': (60,  250),
        'chol':     (80,  700),
        'fbs':      (0,   1),
        'restecg':  (0,   2),
        'thalach':  (50,  220),
        'exang':    (0,   1),
        'oldpeak':  (0,   8),
        'slope':    (0,   2),
        'ca':       (0,   4),
        'thal':     (1,   3),
    }
    range_errors = []
    for feature, (min_val, max_val) in ranges.items():
        val = float(data[feature])
        if not (min_val <= val <= max_val):
            range_errors.append(
                f'{feature} must be between {min_val} and {max_val}, got {val}'
            )
    if range_errors:
        return jsonify({
            'error': 'Value out of expected range',
            'details': range_errors
        }), 400

    # ── Run prediction ────────────────────────────────────────────
    try:
        input_array   = np.array([values])               # shape (1, 13)
        scaled_input  = scaler.transform(input_array)    # standardize
        prediction    = model.predict(scaled_input)[0]   # 0 or 1
        probability   = model.predict_proba(scaled_input)[0][1]  # 0.0 to 1.0

        result = {
            'prediction':   int(prediction),             # 0 = no disease, 1 = disease
            'probability':  round(float(probability), 4),
            'percentage':   round(float(probability) * 100, 1),
            'label':        'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
            'risk_level':   'High'   if probability >= 0.7
                       else 'Medium' if probability >= 0.4
                       else 'Low',
            'input_received': {k: data[k] for k in FEATURE_ORDER}
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

# ── Health check endpoint (Render uses this) ─────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
