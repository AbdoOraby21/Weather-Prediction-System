# 🌧️ ML Capstone Project — Rain in Australia Prediction

A complete Machine Learning pipeline applied to the **WeatherAUS** dataset, covering **Classification**, **Regression**, and **Clustering** tasks.

---

## 📌 Project Goal

- **Classification:** Predict whether it will rain tomorrow (`RainTomorrow`: Yes/No)
- **Regression:** Predict the maximum temperature (`MaxTemp`) for a given day
- **Clustering:** Discover natural weather patterns in Australian cities using K-Means

---

## 📂 Project Structure

```
ML-Capstone-Project/
├── data/
│   └── WeatherAUS.csv
├── notebooks/
│   └── ml_capstone.ipynb
├── results/
│   └── (plots and output figures)
├── project_report.pdf
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Size:** ~145,000 rows × 23 columns
- **Target Variables:**
  - `RainTomorrow` → Classification
  - `MaxTemp` → Regression

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AbdoOraby21/ML-Capstone-Project.git
cd ML-Capstone-Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Add the dataset
Download `WeatherAUS.csv` from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) and place it inside the `data/` folder.

### 5. Run the notebook
```bash
jupyter notebook notebooks/ml_capstone.ipynb
```

---

## 🧪 Models Used

| Task | Model | Key Metric |
|------|-------|------------|
| Classification | Logistic Regression | Accuracy, F1-Score |
| Classification | Decision Tree | Accuracy, Confusion Matrix |
| Classification (Bonus) | Random Forest | Accuracy, Feature Importance |
| Regression | Linear Regression | MSE, RMSE, R² Score |
| Clustering | K-Means (K=3) | Silhouette Score, Elbow Method |

---

## 📈 Key Results

| Model | Result |
|-------|--------|
| Logistic Regression | ~85% Accuracy |
| Decision Tree | ~82% Accuracy |
| Random Forest | ~87% Accuracy |
| Linear Regression | R² ≈ 0.97 |
| K-Means Clustering | Silhouette Score ≈ 0.20+ |

> Exact values depend on the run environment. See `project_report.pdf` for full results.

---

## 📄 Final Report

The full project report is available here: [project_report.pdf](./project_report.pdf)

> 📎 The Google Colab notebook link is included directly inside the PDF report.

---

## 👨‍💻 Author

**Abdelrahman Oraby**
- GitHub: [AbdoOraby21](https://github.com/AbdoOraby21)
- LinkedIn: [abdelrahman-oraby](https://linkedin.com/in/abdelrahman-oraby-92588a2a5)
- Email: orabyabdo21@gmail.com
# ML-Capstone-Project
# Weather-Prediction-System
