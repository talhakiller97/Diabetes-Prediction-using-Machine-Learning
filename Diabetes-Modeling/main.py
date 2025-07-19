import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

#  Load dataset
df = pd.read_csv(r"C:\Users\Talha Saeed\PycharmProjects\week2task3\diabetes.csv")

#  Exploratory Data Analysis (EDA)
print("\n--- Dataset Overview ---")
print(df.head())
print(df.info())
print(df.describe())

# Plot class distribution
sns.countplot(data=df, x='Outcome')
plt.title("Diabetes Class Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Outcome")
plt.show()

# Boxplots by outcome
for col in df.columns[:-1]:
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'{col} vs Outcome')
    plt.show()

# Data Cleaning & Feature Scaling
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

#  Model Training
models = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(max_iter=500)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'F1 Score': f1, 'ROC AUC': auc}

    print(f"\n{name} Results:")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

#  ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

#  SHAP Insights (for Gradient Boosting)
explainer = shap.Explainer(models["Gradient Boosting"], X_train)
shap_values = explainer(X_test, check_additivity=False)
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns)

#  Permutation Feature Importance
perm = permutation_importance(models["Gradient Boosting"], X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm.importances_mean.argsort()

plt.barh(X.columns[sorted_idx], perm.importances_mean[sorted_idx])
plt.title("Permutation Feature Importance")
plt.xlabel("Mean Importance")
plt.show()

# Summary Table
print("\n--- Model Performance Summary ---")
for name, scores in results.items():
    print(f"{name}: F1 Score = {scores['F1 Score']:.3f}, ROC AUC = {scores['ROC AUC']:.3f}")
