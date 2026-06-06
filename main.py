#!/usr/bin/env python
# coding: utf-8

# ============================
# Roman Urdu Hate Speech Classifier (Binary) - Complete
# Meets all project requirements with optimized model selection
# ============================

import pandas as pd
import numpy as np
import re
import string
import time
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

warnings.filterwarnings('ignore')

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv("dataset.csv")
df.columns = ["text", "label"]

print("="*60)
print("ROMAN URDU HATE SPEECH CLASSIFIER")
print("="*60)
print(f"\nDataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['label'].value_counts()}")
print(f"Percentage Hate Speech: {(df['label'].sum()/len(df))*100:.2f}%")

# ============================
# 2. Exploratory Data Analysis
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(data=df, x="label", ax=axes[0])
axes[0].set_title("Label Distribution", fontsize=12)
axes[0].set_xticklabels(["Normal", "Hate Speech"])

df["text_length"] = df["text"].str.len()
sns.boxplot(data=df, x="label", y="text_length", ax=axes[1])
axes[1].set_title("Text Length by Class", fontsize=12)
axes[1].set_xticklabels(["Normal", "Hate Speech"])

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()

# ============================
# 3. Data Preprocessing
# ============================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

initial_rows = len(df)
df = df.drop_duplicates()
df = df.dropna()
print(f"Rows removed (duplicates/null): {initial_rows - len(df)}")

# Roman Urdu Resources
roman_stopwords = set([
    "ka","ki","ke","ko","mein","me","mai","hain","hai","tha","thi","hy",
    "kya","kon","koi","ye","wo","hon","hun","tha","tak","se","to","par",
    "aur","ya","wala","wali","wale","bhi","bhai","agar","magar","na","ne"
])

normalization_map = {
    "boht": "bohot", "bohut": "bohot", "buht": "bohot", "bhut": "bohot",
    "acha": "achha", "gnda": "ganda", "gndu": "gandu", "kamina": "kameena",
    "pagal": "paagal", "lanat": "laanat", "teri": "tumhari", "maa": "mother",
    "behan": "sister", "bhen": "sister", "chutiya": "stupid", "bsdk": "abusive"
}

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_spellings(text):
    for wrong, correct in normalization_map.items():
        text = re.sub(r"\b" + wrong + r"\b", correct, text)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = remove_emojis(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_spellings(text)
    text = " ".join([word for word in text.split() if word not in roman_stopwords])
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ============================
# 4. Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================
# 5. Feature Engineering (TF-IDF)
# ============================
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nFeature space dimension: {X_train_vec.shape[1]}")

# ============================
# 6. Model Definition with Hyperparameter Tuning
# ============================
models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, class_weight='balanced'),
        "params": {
            "C": [0.1, 0.5, 1.0, 2.0],
            "max_iter": [200, 500, 1000],
            "solver": ['liblinear', 'lbfgs']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "SVM": {
        "model": SVC(random_state=42, class_weight='balanced', probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'rbf'],
            "gamma": ['scale', 'auto']
        }
    },
    "Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.1, 0.5, 1.0, 2.0]
        }
    }
}

# ============================
# 7. Model Training & Evaluation
# ============================
results = []

print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

for name, config in models_config.items():
    print(f"\n--- Training {name} ---")

    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_iter=15,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train_vec, y_train)

    train_time = time.time() - start_time

    best_estimator = search.best_estimator_
    y_pred = best_estimator.predict(X_test_vec)
    y_pred_proba = best_estimator.predict_proba(X_test_vec)[:, 1] if hasattr(best_estimator, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    cv_scores = cross_val_score(best_estimator, X_train_vec, y_train, cv=5, scoring='f1')

    results.append({
        "Model": name,
        "Best Params": str(search.best_params_),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "CV Mean F1": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Train Time (s)": train_time,
        "Estimator": best_estimator,
        "y_pred": y_pred,           
        "y_proba": y_pred_proba 
    })

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training Time: {train_time:.2f}s")

# ============================
# 8. Select Best Model (Favor Speed within 1% of best F1)
# ============================
best_f1 = max(r["F1-Score"] for r in results)
threshold = 0.01   # allow up to 1% lower F1

candidates = [r for r in results if r["F1-Score"] >= best_f1 - threshold]
best_result = min(candidates, key=lambda x: x["Train Time (s)"])

best_model = best_result["Estimator"]
best_model_name = best_result["Model"]

print("\n" + "="*60)
print("MODEL SELECTION")
print("="*60)
print(f"Best F1 score among all: {best_f1:.4f}")
print(f"Threshold (≥ {best_f1 - threshold:.4f}): {len(candidates)} model(s) eligible")
for c in candidates:
    print(f"  - {c['Model']}: F1={c['F1-Score']:.4f}, time={c['Train Time (s)']:.2f}s")
print(f"\nSelected {best_model_name} (F1={best_result['F1-Score']:.4f}, time={best_result['Train Time (s)']:.2f}s)")
print("="*60)

# ============================
# 9. Results Comparison Table
# ============================
results_df = pd.DataFrame(results)
display_df = results_df.drop(columns=["Estimator"], errors="ignore")

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
display_df

# ============================
# 9.5 Save Selection Metadata
# ============================
# Add selection info to results_df
results_df['Selected'] = results_df['Model'] == best_model_name
results_df['Selection_Reason'] = "Selected based on F1-Score within 3% threshold, prioritizing speed"

# Create selection metadata dictionary
selection_metadata = {
    'selected_model': best_model_name,
    'selected_f1': best_result['F1-Score'],
    'selected_accuracy': best_result['Accuracy'],
    'selected_precision': best_result['Precision'],
    'selected_recall': best_result['Recall'],
    'selection_criteria': f'F1-Score within {threshold*100}% of best ({best_f1:.4f}), then fastest training time',
    'candidates_considered': len(candidates),
    'candidates_details': [
        {'model': c['Model'], 'f1': c['F1-Score'], 'time': c['Train Time (s)']} 
        for c in candidates
    ],
    'timestamp': datetime.now().isoformat()
}

print("\n" + "="*60)
print("SELECTION METADATA SAVED")
print("="*60)
print(f"Selected Model: {best_model_name}")
print(f"Selection Criteria: {selection_metadata['selection_criteria']}")

# ============================
# 10. Evaluation of Best Model
# ============================
y_pred_best = best_model.predict(X_test_vec)
y_proba_best = best_model.predict_proba(X_test_vec)[:, 1] if hasattr(best_model, "predict_proba") else None

acc_best = accuracy_score(y_test, y_pred_best)
prec_best = precision_score(y_test, y_pred_best)
rec_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
roc_best = roc_auc_score(y_test, y_proba_best) if y_proba_best is not None else None

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)
print(f"Accuracy:  {acc_best:.4f}")
print(f"Precision: {prec_best:.4f}")
print(f"Recall:    {rec_best:.4f}")
print(f"F1-Score:  {f1_best:.4f}")
print(f"ROC-AUC:   {roc_best:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Hate Speech'],
            yticklabels=['Normal', 'Hate Speech'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Hate Speech']))

# ============================
# 11. Feature Importance (if available)
# ============================
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(importances)[-20:]

    plt.figure(figsize=(10,6))
    plt.barh(range(20), importances[top_idx])
    plt.yticks(range(20), [feature_names[i] for i in top_idx])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
elif hasattr(best_model, "coef_"):
    coef = best_model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    top_pos = np.argsort(coef)[-15:]
    top_neg = np.argsort(coef)[:15]

    fig, axes = plt.subplots(1,2, figsize=(14,6))
    axes[0].barh(range(15), coef[top_pos])
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([feature_names[i] for i in top_pos])
    axes[0].set_title('Top Hate Speech Indicators')
    axes[1].barh(range(15), coef[top_neg])
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([feature_names[i] for i in top_neg])
    axes[1].set_title('Top Normal Speech Indicators')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

# ============================
# 10. Bias-Variance Analysis
# ============================
print("\n" + "="*60)
print("BIAS-VARIANCE ANALYSIS")
print("="*60)

for name, config in models_config.items():
    row = results_df[results_df['Model'] == name].iloc[0]
    train_cv_mean = row['CV Mean F1']
    train_cv_std = row['CV Std']
    test_f1 = row['F1-Score']

    bias_estimate = 1 - train_cv_mean
    variance_estimate = train_cv_std

    print(f"\n{name}:")
    print(f"  CV F1 Mean: {train_cv_mean:.4f} ± {train_cv_std:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Bias Estimate: {bias_estimate:.4f}")
    print(f"  Variance Estimate: {variance_estimate:.4f}")

    if abs(train_cv_mean - test_f1) > 0.1:
        print(f"  → Possible overfitting detected!")

# ============================
# 12. Save Model and Vectorizer
# ============================
joblib.dump(best_model, "best_roman_urdu_hate_model.pkl")
joblib.dump(vectorizer, "roman_urdu_vectorizer.pkl")
joblib.dump(results_df, "model_results.pkl")
joblib.dump(selection_metadata, "selection_metadata.pkl")  # Save selection info separately

print("\n" + "="*60)
print("Model and vectorizer saved successfully!")
print(f"Best model: {best_model_name}")
print(f"Final F1-Score: {f1_best:.4f}")
print(f"Selection metadata saved to: selection_metadata.pkl")
print("="*60)

# ============================
# ROC Curves for All Models
# ============================
plt.figure(figsize=(10, 8))

for res in results:
    name = res['Model']
    y_proba = res['y_proba']
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig("roc_curves.png", dpi=150)
plt.show()


# Metrics to plot
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']
# Time is separate because scale is different

# Prepare data
model_names = results_df['Model'].values
metric_values = {m: results_df[m].values for m in metrics}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plot Precision, Recall, F1-Score, Accuracy on first three subplots
for i, metric in enumerate(metrics[:4]):
    ax = axes[i]
    bars = ax.bar(model_names, metric_values[metric], color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.tick_params(axis='x', rotation=15)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Plot ROC-AUC (fourth plot, but we have 4 subplots already)
ax = axes[3]
bars = ax.bar(model_names, metric_values['ROC-AUC'], color='seagreen')
ax.set_ylim(0, 1)
ax.set_ylabel('ROC-AUC')
ax.set_title('ROC-AUC Comparison')
ax.tick_params(axis='x', rotation=15)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison_metrics.png", dpi=150)
plt.show()

# Plot Training Time separately (log scale because SVM is huge)
plt.figure(figsize=(10, 5))
train_times = results_df['Train Time (s)'].values
bars = plt.bar(model_names, train_times, color='coral')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.yscale('log')  # Use log scale to see differences clearly
plt.tick_params(axis='x', rotation=15)
for bar, t in zip(bars, train_times):
    plt.annotate(f'{t:.2f}s', xy=(bar.get_x() + bar.get_width()/2, t),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison_time.png", dpi=150)
plt.show()