# 🩺 Diabetes Prediction using Machine Learning

This project aims to predict diabetes presence using machine learning models and interpretability techniques. It involves detailed exploratory data analysis, model evaluation (F1, ROC-AUC), and feature importance interpretation using SHAP and permutation methods.

---

## 📌 Project Objectives

- Perform exploratory data analysis (EDA) on the diabetes dataset.
- Handle missing or zero values in medical measurements.
- Train and evaluate three different machine learning models.
- Use SHAP and permutation importance to explain model predictions.
- Compare model performance using F1 Score and ROC-AUC.

---

## 📁 Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Format:** CSV
- **Source:** [Local Path] `diabetes.csv`
- **Target Variable:** `Outcome` (0 = No Diabetes, 1 = Diabetes)

---

## 🧰 Tools & Libraries

- **Programming Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `seaborn`, `matplotlib`
  - `scikit-learn`
  - `shap`
  - `warnings`

---

## 📊 Exploratory Data Analysis (EDA)

- Class distribution using count plot
- Feature correlation heatmap
- Boxplots for each feature by outcome class

---

## 🧼 Data Preprocessing

- Replaced invalid 0s with NaN in key features: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- Filled missing values with median
- Scaled features using `StandardScaler`

---

## 🧪 Machine Learning Models

Three classifiers were trained and compared:

| Model              | Description                     |
|-------------------|---------------------------------|
| 🎯 Gradient Boosting | High accuracy, interpretable     |
| ⚙️ Support Vector Machine | Good for small/medium-sized data |
| 🧠 Neural Network      | Capable of learning complex patterns |

---

## 📈 Evaluation Metrics

- ✅ F1 Score
- 📈 ROC-AUC Score
- 📉 ROC Curve
- 📜 Classification Report

---

## 🔍 Model Interpretability

### 💡 SHAP Summary Plot

Used SHAP to explain feature importance from the **Gradient Boosting** model:

```python
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
📊 Permutation Feature Importance
Ranked features based on their impact on predictions:


perm = permutation_importance(model, X_test, y_test, n_repeats=10)
🗂️ Project Structure

├── diabetes_prediction.py        # Main script
├── diabetes.csv                  # Dataset
├── README.md                     # Documentation
▶️ How to Run
Clone the repository


git clone https://github.com/yourusername/diabetes-prediction-ml.git
cd diabetes-prediction-ml
Install dependencies


pip install -r requirements.txt
Run the script


python diabetes_prediction.py
📊 Sample Output
java

--- Model Performance Summary ---
Gradient Boosting: F1 Score = 0.782, ROC AUC = 0.876
SVM: F1 Score = 0.756, ROC AUC = 0.849
Neural Network: F1 Score = 0.764, ROC AUC = 0.863
👤 Author
Talha Saeed
📍 Data Scientist
