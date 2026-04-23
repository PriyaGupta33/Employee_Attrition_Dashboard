# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:43:49 2026

@author: HP
"""

# ============================================
# EMPLOYEE ATTRITION ANALYSIS AND PREDICTION
# Spyder-friendly full capstone code
# Dataset: HR_comma_sep.csv
# Target: left
# ============================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv("P:\sem 6\combinatorial\project\HR_comma_sep.csv")
print("Dataset loaded successfully.\n")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# -----------------------------
# 2. BASIC INFORMATION
# -----------------------------
print("\n" + "=" * 60)
print("STEP 2: DATASET INFORMATION")
print("=" * 60)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates if any
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# -----------------------------
# 3. RENAME COLUMNS FOR BETTER READABILITY
# -----------------------------
print("\n" + "=" * 60)
print("STEP 3: RENAMING COLUMNS")
print("=" * 60)

df.rename(columns={
    'satisfaction_level': 'satisfaction_level',
    'last_evaluation': 'last_evaluation',
    'number_project': 'number_project',
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'time_spend_company',
    'Work_accident': 'work_accident',
    'promotion_last_5years': 'promotion_last_5years',
    'Department': 'department',
    'left': 'attrition'
}, inplace=True)

print("Updated columns:")
print(df.columns.tolist())

# -----------------------------
# 4. TARGET ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("STEP 4: TARGET VARIABLE ANALYSIS")
print("=" * 60)

print("\nAttrition value counts:")
print(df['attrition'].value_counts())

print("\nAttrition percentage:")
print(df['attrition'].value_counts(normalize=True) * 100)

# -----------------------------
# 5. EXPLORATORY DATA ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("STEP 5: EDA")
print("=" * 60)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x='attrition', data=df, palette='Set2')
plt.title("Attrition Count")
plt.xlabel("Attrition (0 = Stay, 1 = Leave)")
plt.ylabel("Count")
plt.show()

# Salary vs Attrition
plt.figure(figsize=(7, 5))
sns.countplot(x='salary', hue='attrition', data=df, palette='Set1')
plt.title("Salary vs Attrition")
plt.show()

# Department vs Attrition
plt.figure(figsize=(12, 5))
sns.countplot(x='department', hue='attrition', data=df, palette='coolwarm')
plt.title("Department vs Attrition")
plt.xticks(rotation=45)
plt.show()

# Distribution plots
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('attrition')

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots for outlier visualization
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot (smaller subset for visibility)
pairplot_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                 'average_monthly_hours', 'time_spend_company', 'attrition']
sns.pairplot(df[pairplot_cols], hue='attrition', palette='husl')
plt.show()

# -----------------------------
# 6. OUTLIER HANDLING (IQR METHOD)
# -----------------------------
print("\n" + "=" * 60)
print("STEP 6: OUTLIER DETECTION")
print("=" * 60)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"{col}: {outliers} outliers")

print("\nNote: Outliers are not removed aggressively because they may carry business meaning in HR data.")

# -----------------------------
# 7. FEATURE ENGINEERING
# -----------------------------
print("\n" + "=" * 60)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 60)

X = df.drop('attrition', axis=1)
y = df['attrition']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess training data separately for SMOTE
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix if needed
try:
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()
except:
    pass

print("Training data shape after preprocessing:", X_train_processed.shape)
print("Testing data shape after preprocessing:", X_test_processed.shape)

# Handle imbalance using SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print("Shape before SMOTE:", X_train_processed.shape, y_train.shape)
print("Shape after SMOTE:", X_train_smote.shape, y_train_smote.shape)

# -----------------------------
# 8. MODEL BUILDING
# -----------------------------
print("\n" + "=" * 60)
print("STEP 8: MODEL BUILDING")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
}

results = []
best_model = None
best_auc = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, prec, rec, f1, auc])

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# -----------------------------
# 9. MODEL COMPARISON TABLE
# -----------------------------
print("\n" + "=" * 60)
print("STEP 9: MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"
])

results_df = results_df.sort_values(by="AUC Score", ascending=False)
print(results_df)

# Plot model comparison
plt.figure(figsize=(10, 5))
sns.barplot(x="AUC Score", y="Model", data=results_df, palette="viridis")
plt.title("Model Comparison based on AUC Score")
plt.show()

print(f"\nBest Model: {best_model_name}")
print(f"Best AUC Score: {best_auc:.4f}")

# -----------------------------
# 10. EVALUATE BEST MODEL
# -----------------------------
print("\n" + "=" * 60)
print("STEP 10: BEST MODEL EVALUATION")
print("=" * 60)

y_pred_best = best_model.predict(X_test_processed)
y_prob_best = best_model.predict_proba(X_test_processed)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {best_auc:.4f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -----------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------
print("\n" + "=" * 60)
print("STEP 11: FEATURE IMPORTANCE")
print("=" * 60)

feature_names = preprocessor.get_feature_names_out()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
    plt.title(f"Top 15 Important Features - {best_model_name}")
    plt.show()

elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
    plt.title(f"Top 15 Important Features - {best_model_name}")
    plt.show()

else:
    print("Feature importance not available for this model.")

# -----------------------------
# 12. SAVE BEST MODEL
# -----------------------------
print("\n" + "=" * 60)
print("STEP 12: SAVING BEST MODEL")
print("=" * 60)

joblib.dump(best_model, "best_attrition_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("Best model saved as best_attrition_model.pkl")
print("Preprocessor saved as preprocessor.pkl")

# -----------------------------
# 13. SAMPLE PREDICTION
# -----------------------------
print("\n" + "=" * 60)
print("STEP 13: SAMPLE PREDICTION")
print("=" * 60)

sample_employee = X.iloc[[0]]
sample_processed = preprocessor.transform(sample_employee)

try:
    sample_processed = sample_processed.toarray()
except:
    pass

prediction = best_model.predict(sample_processed)[0]
probability = best_model.predict_proba(sample_processed)[0][1]

print("Sample employee prediction:")
print("Predicted Attrition:", prediction)
print("Probability of Leaving:", round(probability, 4))

if prediction == 1:
    print("This employee is likely to leave.")
else:
    print("This employee is likely to stay.")

print("\nProject execution completed successfully.")