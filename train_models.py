# =========================
# train_models.py
# Run this ONCE to train and save all models
# Usage: python train_models.py
# =========================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

# ─── Load Data ────────────────────────────────────────────────────────────────
print("[1/7] Loading data...")
df = pd.read_csv("data/WeatherAUS.csv")

# ─── Drop High-Missing Columns ────────────────────────────────────────────────
print("[2/7] Preprocessing...")
df.drop(columns=['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm'], inplace=True)
df = df.dropna(subset=['RainTomorrow'])

# Date Engineering
df['Date']  = pd.to_datetime(df['Date'])
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day']   = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)

# Encode Binary
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df['RainToday']    = df['RainToday'].map({'No': 0, 'Yes': 1})

# Fill Missing
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-Hot Encode
df = pd.get_dummies(df, drop_first=True)

# Save column names for the Flask app to use
feature_cols = [c for c in df.columns if c not in ['RainTomorrow', 'MaxTemp']]

# ─── Classification ───────────────────────────────────────────────────────────
print("[3/7] Training Logistic Regression (Classification)...")
X_cls = df.drop('RainTomorrow', axis=1)
y_cls = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_cls = StandardScaler()
X_train_s  = scaler_cls.fit_transform(X_train)
X_test_s   = scaler_cls.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_s, y_train)
cls_accuracy = accuracy_score(y_test, log_model.predict(X_test_s))
print(f"    → Classification Accuracy: {cls_accuracy:.4f}")

# ─── Regression ───────────────────────────────────────────────────────────────
print("[4/7] Training Linear Regression...")
X_reg = df.drop(['MaxTemp', 'RainTomorrow'], axis=1, errors='ignore')
y_reg = df['MaxTemp']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
r2 = r2_score(y_test_r, lr_model.predict(X_test_r))
print(f"    → Regression R² Score: {r2:.4f}")


# ─── Clustering ───────────────────────────────────────────────────────────────
print("[5/7] Training K-Means (K=3)...")

df_cluster = df.drop('RainTomorrow', axis=1)

scaler_cl = StandardScaler()
X_scaled = scaler_cl.fit_transform(df_cluster)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Calculate Silhouette Score on a sample to avoid huge computation
sample_size = min(5000, len(X_scaled))

sample_idx = np.random.choice(
    len(X_scaled),
    sample_size,
    replace=False
)

sil = silhouette_score(
    X_scaled[sample_idx],
    clusters[sample_idx]
)

print(f"    → Silhouette Score: {sil:.4f}")
# ─── Save Models ──────────────────────────────────────────────────────────────
print("[6/7] Saving models...")
os.makedirs("models", exist_ok=True)

with open("models/logistic_model.pkl",  "wb") as f: pickle.dump(log_model,  f)
with open("models/scaler_cls.pkl",      "wb") as f: pickle.dump(scaler_cls, f)
with open("models/linear_model.pkl",    "wb") as f: pickle.dump(lr_model,   f)
with open("models/kmeans_model.pkl",    "wb") as f: pickle.dump(kmeans,     f)
with open("models/scaler_cl.pkl",       "wb") as f: pickle.dump(scaler_cl,  f)
with open("models/feature_cols.pkl",    "wb") as f: pickle.dump(list(X_cls.columns), f)
with open("models/reg_feature_cols.pkl","wb") as f: pickle.dump(list(X_reg.columns), f)
with open("models/cluster_cols.pkl",    "wb") as f: pickle.dump(list(df_cluster.columns), f)

# Save metrics for display in the app
metrics = {
    "cls_accuracy": round(cls_accuracy * 100, 2),
    "r2_score":     round(r2, 4),
    "rmse":         round(float(np.sqrt(((y_test_r - lr_model.predict(X_test_r))**2).mean())), 4),
    "silhouette":   round(sil, 4),
}
with open("models/metrics.pkl", "wb") as f: pickle.dump(metrics, f)

print("[7/7] Done! All models saved in /models folder.")
print(f"\n  Summary:")
print(f"  Classification Accuracy : {cls_accuracy*100:.2f}%")
print(f"  Regression R² Score     : {r2:.4f}")
print(f"  Clustering Silhouette   : {sil:.4f}")
