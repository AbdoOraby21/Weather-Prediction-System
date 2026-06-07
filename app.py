# =========================
# app.py — Flask Web Application
# Run: python app.py
# =========================

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ─── Load Models ──────────────────────────────────────────────────────────────
def load_models():
    models = {}
    try:
        with open("models/logistic_model.pkl",   "rb") as f: models["log"]         = pickle.load(f)
        with open("models/scaler_cls.pkl",        "rb") as f: models["scaler_cls"]  = pickle.load(f)
        with open("models/linear_model.pkl",      "rb") as f: models["lr"]          = pickle.load(f)
        with open("models/kmeans_model.pkl",      "rb") as f: models["kmeans"]      = pickle.load(f)
        with open("models/scaler_cl.pkl",         "rb") as f: models["scaler_cl"]   = pickle.load(f)
        with open("models/feature_cols.pkl",      "rb") as f: models["cls_cols"]    = pickle.load(f)
        with open("models/reg_feature_cols.pkl",  "rb") as f: models["reg_cols"]    = pickle.load(f)
        with open("models/cluster_cols.pkl",      "rb") as f: models["cluster_cols"]= pickle.load(f)
        with open("models/metrics.pkl",           "rb") as f: models["metrics"]     = pickle.load(f)
        print("✓ All models loaded successfully.")
    except FileNotFoundError:
        print("✗ Models not found. Please run: python train_models.py")
    return models

models = load_models()

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    metrics = models.get("metrics", {})
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Build input row ──────────────────────────────────────────────────
        raw = {
            "MinTemp":       float(data.get("MinTemp", 15)),
            "MaxTemp":       float(data.get("MaxTemp", 25)),
            "Rainfall":      float(data.get("Rainfall", 0)),
            "WindGustSpeed": float(data.get("WindGustSpeed", 40)),
            "WindSpeed9am":  float(data.get("WindSpeed9am", 15)),
            "WindSpeed3pm":  float(data.get("WindSpeed3pm", 20)),
            "Humidity9am":   float(data.get("Humidity9am", 70)),
            "Humidity3pm":   float(data.get("Humidity3pm", 50)),
            "Pressure9am":   float(data.get("Pressure9am", 1015)),
            "Pressure3pm":   float(data.get("Pressure3pm", 1012)),
            "Temp9am":       float(data.get("Temp9am", 18)),
            "Temp3pm":       float(data.get("Temp3pm", 23)),
            "RainToday":     int(data.get("RainToday", 0)),
            "Year":          int(data.get("Year", 2015)),
            "Month":         int(data.get("Month", 6)),
            "Day":           int(data.get("Day", 15)),
        }

        # Location & wind direction dummies (default all 0 — unknown)
        location = data.get("Location", "")
        wind_gust_dir = data.get("WindGustDir", "")
        wind_dir_9am  = data.get("WindDir9am", "")
        wind_dir_3pm  = data.get("WindDir3pm", "")

        # ── Classification ───────────────────────────────────────────────────
        cls_cols   = models["cls_cols"]
        input_df   = pd.DataFrame([{c: 0 for c in cls_cols}])
        for k, v in raw.items():
            if k in input_df.columns:
                input_df[k] = v
        # Set location/wind dummies
        for col in cls_cols:
            if location and col == f"Location_{location}":
                input_df[col] = 1
            if wind_gust_dir and col == f"WindGustDir_{wind_gust_dir}":
                input_df[col] = 1
            if wind_dir_9am and col == f"WindDir9am_{wind_dir_9am}":
                input_df[col] = 1
            if wind_dir_3pm and col == f"WindDir3pm_{wind_dir_3pm}":
                input_df[col] = 1

        X_cls_scaled = models["scaler_cls"].transform(input_df)
        cls_pred     = int(models["log"].predict(X_cls_scaled)[0])
        cls_proba    = float(models["log"].predict_proba(X_cls_scaled)[0][1])

        # ── Regression ───────────────────────────────────────────────────────
        reg_cols  = models["reg_cols"]
        reg_df    = pd.DataFrame([{c: 0 for c in reg_cols}])
        for k, v in raw.items():
            if k in reg_df.columns and k != "MaxTemp":
                reg_df[k] = v
        for col in reg_cols:
            if location and col == f"Location_{location}":
                reg_df[col] = 1
        reg_pred = float(models["lr"].predict(reg_df)[0])

        # ── Clustering ───────────────────────────────────────────────────────
        cl_cols   = models["cluster_cols"]
        cl_df     = pd.DataFrame([{c: 0 for c in cl_cols}])
        for k, v in raw.items():
            if k in cl_df.columns:
                cl_df[k] = v
        for col in cl_cols:
            if location and col == f"Location_{location}":
                cl_df[col] = 1
        X_cl_scaled = models["scaler_cl"].transform(cl_df)
        cluster_id  = int(models["kmeans"].predict(X_cl_scaled)[0])

        cluster_names = {0: "Hot & Dry", 1: "Mild", 2: "Cool & Wet"}

        return jsonify({
            "success": True,
            "classification": {
                "prediction": cls_pred,
                "label":      "Rain 🌧️" if cls_pred == 1 else "No Rain ☀️",
                "probability": round(cls_proba * 100, 1),
            },
            "regression": {
                "predicted_maxtemp": round(reg_pred, 1),
            },
            "clustering": {
                "cluster_id":   cluster_id,
                "cluster_name": cluster_names.get(cluster_id, "Unknown"),
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
