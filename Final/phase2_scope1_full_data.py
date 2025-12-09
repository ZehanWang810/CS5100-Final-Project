from student_project import load_data, preprocess_data, train_gb_pipeline, RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

# ---------- 1) 用 mini dataset（Phase 1 的设置）做基线 ----------
df_mini = load_data("datasets/student-mat-mini.csv")
proc_mini = preprocess_data(df_mini)
X_mini = proc_mini.drop(columns=["at_risk"]).values
y_mini = proc_mini["at_risk"].values

Xtr_mi, Xte_mi, ytr_mi, yte_mi = train_test_split(
    X_mini, y_mini, test_size=0.3, random_state=42, stratify=y_mini
)

gb_mini = train_gb_pipeline(Xtr_mi, ytr_mi)
y_pred_gb_mi = gb_mini.predict(Xte_mi)
y_prob_gb_mi = gb_mini.predict_proba(Xte_mi)[:, 1]

print("=== Gradient Boosting on MINI dataset ===")
print("F1:", f1_score(yte_mi, y_pred_gb_mi))
print("ROC-AUC:", roc_auc_score(yte_mi, y_prob_gb_mi))

rf_mini = RandomForest(n_estimators=5, max_depth=4,
                       sample_size=min(64, len(Xtr_mi)), random_state=42)
rf_mini.fit(Xtr_mi, ytr_mi)
y_pred_rf_mi = rf_mini.predict(Xte_mi)
print("Random Forest F1 (mini):", f1_score(yte_mi, y_pred_rf_mi))

print("\n----------------------------------------\n")

# ---------- 2) 用 FULL dataset 做 Phase 2 的扩展 ----------
df_full = load_data("datasets/student-mat.csv")
proc_full = preprocess_data(df_full)
X_full = proc_full.drop(columns=["at_risk"]).values
y_full = proc_full["at_risk"].values

Xtr, Xte, ytr, yte = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
)

gb_full = train_gb_pipeline(Xtr, ytr)
y_pred_gb_full = gb_full.predict(Xte)
y_prob_gb_full = gb_full.predict_proba(Xte)[:, 1]

print("=== Gradient Boosting on FULL dataset ===")
print("F1:", f1_score(yte, y_pred_gb_full))
print("ROC-AUC:", roc_auc_score(yte, y_prob_gb_full))

rf_full = RandomForest(n_estimators=5, max_depth=4,
                       sample_size=min(64, len(Xtr)), random_state=42)
rf_full.fit(Xtr, ytr)
y_pred_rf_full = rf_full.predict(Xte)
print("Random Forest F1 (full):", f1_score(yte, y_pred_rf_full))
