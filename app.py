from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap

# Load models
binary_model = joblib.load("binary_fault_model.joblib")
multi_model = joblib.load("multiclass_error_model.joblib")
encoder = joblib.load("errorcode_label_encoder.joblib")

# Feature order
features = [
    "Temperature(°C)", "Pressure(bar)", "FlowRate(L/min)",
    "Vibration(mm/s)", "FillHeight(mm)", "PowerConsumption(kW)",
    "CO2_Level(ppm)", "Humidity(%)"
]

# Error descriptions
error_descriptions = {
    "E000": "No fault detected — all system parameters are within the normal operating range.",
    "E001": "Over Temperature Detected — the temperature has exceeded the acceptable threshold, indicating possible cooling failure or excessive load.",
    "E002": "High Pressure Condition — the system pressure is above the safe limit, which may indicate blockage, pump malfunction, or valve issues.",
    "E003": "Flow Rate Out of Range — the flow value is either too low or too high, suggesting possible leakage, clogging, or pump irregularities.",
    "E004": "High Vibration Detected — abnormal vibrations were recorded, which may signal imbalance, bearing wear, or mechanical component misalignment.",
    "E005": "Fill Height Abnormal — the tank/container level is outside the expected range, possibly due to sensor error, leakage, or improper filling.",
    "E006": "Power Consumption Abnormal — the energy usage is inconsistent with normal operation, indicating overload, electrical issues, or equipment deterioration.",
    "E007": "CO2 Level Out of Range — carbon dioxide concentration has deviated from the standard range, potentially affecting product quality or safety.",
    "E008": "Humidity Out of Range — humidity levels are abnormal, which may cause condensation, corrosion, or impact on product stability."
}

app = Flask(__name__)

# === SHAP EXPLAINER LOADING ===
try:
    shap_explainer = shap.TreeExplainer(binary_model)
except Exception:
    shap_explainer = shap.Explainer(binary_model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}

        # Safe conversion
        def safe_float(val):
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
            return float(val)

        # Collect input values
        input_keys = ['temp','pressure','flow','vibration','fillheight','power','co2','humidity']
        input_values = []
        for key in input_keys:
            val = data.get(key, 0)
            try:
                val = safe_float(val)
            except:
                val = 0
            input_values.append(val)

        # Create DF
        df = pd.DataFrame([input_values], columns=features)

        # === BINARY MODEL PREDICTION ===
        binary_prob = float(binary_model.predict_proba(df)[0][1])
        binary_pred = int(binary_model.predict(df)[0])

        # === MULTICLASS ERROR CODE PREDICTION ===
        multi_raw = int(multi_model.predict(df)[0])
        error_code = encoder.inverse_transform([multi_raw])[0]

        # Override to normal if probability extremely low
        if binary_prob < 0.05:
            error_code = "E000"

        # === SHAP CONTRIBUTIONS ===
        try:
            shap_values = shap_explainer.shap_values(df)
            shap_values = np.array(shap_values)

            # For binary class models returning (1, n_features, 2)
            if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]

            shap_flat = shap_values.ravel()[:len(features)]
            contributions = {
                features[i]: float(shap_flat[i]) for i in range(len(features))
            }

        except Exception as e:
            print("SHAP error:", e)
            contributions = {f: 0 for f in features}

        # Print terminal logs
        print("\n--- /predict called ---")
        for f, v in zip(features, input_values):
            print(f"{f}: {v}")
        print(f"Binary: {binary_pred}, Prob: {binary_prob}")
        print(f"Error Code: {error_code}")
        print("SHAP:", contributions)
        print("----------------------\n")

        # Return response
        return jsonify({
            "binary": binary_pred,
            "probability": round(binary_prob, 4),
            "error_code": error_code,
            "error_description": error_descriptions.get(error_code, "No fault detected - normal."),
            "contributions": contributions
        })

    except Exception as e:
        return jsonify({
            "binary": 0,
            "probability": 0,
            "error_code": "E000",
            "error_description": "No fault detected - normal.",
            "contributions": {},
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)