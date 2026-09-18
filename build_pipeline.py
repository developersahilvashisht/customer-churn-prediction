"""
Customer Churn Prediction - End-to-End Build Script
=====================================================
This script performs the full workflow and is also the source of truth
for the Jupyter notebook.
Run: python3 build_pipeline.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from feature_engineering import engineer_features

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110
RANDOM_STATE = 42
FIG_DIR = "notebook/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. DATA UNDERSTANDING & PREPARATION
# ============================================================
print("="*60, "\n1. DATA UNDERSTANDING & PREPARATION\n", "="*60)

df = pd.read_csv("data/TelcoCustomerChurn.csv")
print("Shape:", df.shape)

# --- Fix TotalCharges: stored as string, 11 rows are blank spaces for
# brand-new customers (tenure == 0), so they were never billed yet.
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
n_missing_totalcharges = df['TotalCharges'].isnull().sum()
print(f"Missing TotalCharges (all tenure=0 new customers): {n_missing_totalcharges}")
# These customers have tenure=0 -> total charges should logically be 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Duplicate check
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customerIDs:", df['customerID'].duplicated().sum())

# Drop identifier column - not predictive
df = df.drop(columns=['customerID'])

# SeniorCitizen is 0/1 but conceptually categorical -> map to Yes/No for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\nChurn distribution:\n", df['Churn'].value_counts(normalize=True).round(3))

# Identify numerical vs categorical
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [c for c in df.columns if c not in numerical_features + ['Churn']]
print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "="*60, "\n2. EXPLORATORY DATA ANALYSIS\n", "="*60)

# --- Plot 1: Churn distribution
fig, ax = plt.subplots(figsize=(5,4))
counts = df['Churn'].value_counts().sort_index()
labels = ['No Churn', 'Churn']
colors = ['#4C72B0', '#DD8452']
ax.bar(labels, counts.values, color=colors)
for i, v in enumerate(counts.values):
    ax.text(i, v+50, f"{v} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
ax.set_title("Churn Distribution")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_churn_distribution.png")
plt.close()

# --- Plot 2: Churn by Contract type
fig, ax = plt.subplots(figsize=(6,4))
ct = pd.crosstab(df['Contract'], df['Churn'], normalize='index')*100
ct.columns = ['No Churn', 'Churn']
ct.plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title("Churn Rate by Contract Type")
ax.set_ylabel("% of Customers")
ax.set_xlabel("Contract Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_churn_by_contract.png")
plt.close()

# --- Plot 3: Churn by Internet Service
fig, ax = plt.subplots(figsize=(6,4))
ct2 = pd.crosstab(df['InternetService'], df['Churn'], normalize='index')*100
ct2.columns = ['No Churn', 'Churn']
ct2.plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title("Churn Rate by Internet Service Type")
ax.set_ylabel("% of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_churn_by_internet.png")
plt.close()

# --- Plot 4: Tenure distribution by churn
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, multiple='stack',
             palette=colors, ax=ax)
ax.set_title("Tenure Distribution by Churn Status")
ax.set_xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_tenure_by_churn.png")
plt.close()

# --- Plot 5: Monthly Charges distribution by churn
fig, ax = plt.subplots(figsize=(6,4))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True,
            palette=colors, ax=ax, common_norm=False, alpha=0.4)
ax.set_title("Monthly Charges Distribution by Churn Status")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_monthlycharges_by_churn.png")
plt.close()

# --- Plot 6: Churn by Payment Method
fig, ax = plt.subplots(figsize=(7,4))
ct3 = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')*100
ct3.columns = ['No Churn', 'Churn']
ct3.sort_values('Churn', ascending=False).plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title("Churn Rate by Payment Method")
ax.set_ylabel("% of Customers")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_churn_by_payment.png")
plt.close()

# --- Plot 7: Correlation heatmap of numerical features
fig, ax = plt.subplots(figsize=(5,4))
corr = df[numerical_features + ['Churn']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title("Correlation: Numerical Features & Churn")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_correlation_heatmap.png")
plt.close()

print(f"Saved 7 EDA figures to {FIG_DIR}/")

# Business insight summary
insights = {
    "churn_rate": f"{df['Churn'].mean()*100:.1f}%",
    "month_to_month_churn": f"{df[df.Contract=='Month-to-month'].Churn.mean()*100:.1f}%",
    "two_year_churn": f"{df[df.Contract=='Two year'].Churn.mean()*100:.1f}%",
    "fiber_churn": f"{df[df.InternetService=='Fiber optic'].Churn.mean()*100:.1f}%",
    "electronic_check_churn": f"{df[df.PaymentMethod=='Electronic check'].Churn.mean()*100:.1f}%",
    "median_tenure_churned": df[df.Churn==1].tenure.median(),
    "median_tenure_retained": df[df.Churn==0].tenure.median(),
}
print("\nKey insights:", json.dumps(insights, indent=2))

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60, "\n3. FEATURE ENGINEERING\n", "="*60)

# NOTE: engineer_features() is imported from feature_engineering.py so that
# training and the API (app.py) always apply IDENTICAL transformations to
# raw data. See that file for the rationale behind each feature.
df = engineer_features(df)
print("New features added: tenure_group, num_services, avg_monthly_spend, charge_increase")
print(df[['tenure_group', 'num_services', 'avg_monthly_spend', 'charge_increase']].head())

numerical_features = numerical_features + ['num_services', 'avg_monthly_spend', 'charge_increase']
categorical_features = categorical_features + ['tenure_group']

# ============================================================
# TRAIN/TEST SPLIT (before any fitting -> avoid leakage)
# ============================================================
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train churn rate:", y_train.mean().round(3), "| Test churn rate:", y_test.mean().round(3))

# ============================================================
# PREPROCESSING PIPELINE (fit ONLY on train, applied consistently to test/new data)
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ============================================================
# 4. MODEL DEVELOPMENT - Decision Tree, 2+ configurations
# ============================================================
print("\n" + "="*60, "\n4. MODEL DEVELOPMENT\n", "="*60)

configs = {
    "DT_default_depth5": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "DT_deeper_balanced": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20, class_weight='balanced', random_state=RANDOM_STATE
    ),
    "DT_shallow_gini_minsplit": DecisionTreeClassifier(
        max_depth=4, min_samples_split=50, criterion='gini', random_state=RANDOM_STATE
    ),
}

results = {}
fitted_pipelines = {}

for name, clf in configs.items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    fitted_pipelines[name] = pipe
    print(f"\n{name}: {results[name]}")

results_df = pd.DataFrame(results).T.round(4)
print("\nModel comparison:\n", results_df)

# Select final model: best F1 (balances precision & recall) - business-justified below
final_model_name = results_df['f1'].idxmax()
print(f"\nSelected final model: {final_model_name} (highest F1 score)")
final_pipeline = fitted_pipelines[final_model_name]

# ============================================================
# 5. MODEL EVALUATION
# ============================================================
print("\n" + "="*60, "\n5. MODEL EVALUATION\n", "="*60)

y_pred_final = final_pipeline.predict(X_test)
y_proba_final = final_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_final)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {final_model_name}")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_confusion_matrix.png")
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(5,4))
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba_final):.3f}")
ax.plot([0,1],[0,1],'--',color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Final Model")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_roc_curve.png")
plt.close()

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(7,4))
results_df[['accuracy','precision','recall','f1']].plot(kind='bar', ax=ax)
ax.set_title("Model Configuration Comparison")
ax.set_ylabel("Score")
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_model_comparison.png")
plt.close()

# ============================================================
# 6. MODEL INTERPRETATION
# ============================================================
print("\n" + "="*60, "\n6. MODEL INTERPRETATION\n", "="*60)

feature_names = final_pipeline.named_steps['preprocess'].get_feature_names_out()
importances = final_pipeline.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).head(15)
print(fi_df)

fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(data=fi_df, y='feature', x='importance', ax=ax, color='#4C72B0')
ax.set_title(f"Top 15 Feature Importances - {final_model_name}")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_feature_importance.png")
plt.close()

# Decision tree visualization (limited depth for readability)
fig, ax = plt.subplots(figsize=(20,10))
plot_tree(final_pipeline.named_steps['model'], max_depth=3, feature_names=feature_names,
          class_names=['No Churn', 'Churn'], filled=True, fontsize=8, ax=ax)
ax.set_title(f"Decision Tree Structure (top 3 levels) - {final_model_name}")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_tree_visualization.png")
plt.close()

print(f"Saved interpretation figures to {FIG_DIR}/")

# ============================================================
# 7. SAVE MODEL & ARTIFACTS
# ============================================================
print("\n" + "="*60, "\n7. MODEL SAVING\n", "="*60)

os.makedirs("model", exist_ok=True)
joblib.dump(final_pipeline, "model/churn_model.pkl")
print("Saved final pipeline to model/churn_model.pkl")

# Save metadata needed by the API for input validation
metadata = {
    "numerical_features_raw": ['tenure', 'MonthlyCharges', 'TotalCharges'],
    "categorical_features_raw": [c for c in categorical_features if c != 'tenure_group'],
    "engineered": ["tenure_group", "num_services", "avg_monthly_spend", "charge_increase"],
    "final_model": final_model_name,
    "model_metrics": results[final_model_name],
    "all_model_results": results,
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print("Saved model/metadata.json")

# Save results_df and insights for notebook/report use
results_df.to_csv("model/model_comparison.csv")
with open("model/eda_insights.json", "w") as f:
    json.dump(insights, f, indent=2, default=str)

print("\nDONE.")
