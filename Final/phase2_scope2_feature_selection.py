from student_project import load_data, preprocess_data, train_gb_pipeline, RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np


# ---------- 0) 载入 FULL 数据并预处理 ----------
df_full = load_data("datasets/student-mat.csv")
proc_full = preprocess_data(df_full)

X_full_df = proc_full.drop(columns=["at_risk"])
y_full = proc_full["at_risk"].values

print("Full feature matrix shape:", X_full_df.shape)  # (n_samples, n_features)


# ---------- 1) 计算每个特征与 at_risk 之间的互信息（importance） ----------
mi = mutual_info_classif(X_full_df.values, y_full, random_state=42)
mi_series = pd.Series(mi, index=X_full_df.columns).sort_values(ascending=False)

print("\nTop 20 features by mutual information with 'at_risk':")
print(mi_series.head(20))


# ---------- 2) 选择前 K 个特征 ----------
TOP_K = 15
top_features = mi_series.head(TOP_K).index.tolist()

print(f"\nUsing top {TOP_K} features:")
print(top_features)


# ---------- 3) 基于“全部特征”的基线（与 Scope 1 一致） ----------
X_all = X_full_df.values
Xtr_all, Xte_all, ytr_all, yte_all = train_test_split(
    X_all, y_full, test_size=0.3, random_state=42, stratify=y_full
)

gb_all = train_gb_pipeline(Xtr_all, ytr_all)
y_pred_gb_all = gb_all.predict(Xte_all)
y_prob_gb_all = gb_all.predict_proba(Xte_all)[:, 1]

rf_all = RandomForest(n_estimators=5, max_depth=4,
                      sample_size=min(64, len(Xtr_all)), random_state=42)
rf_all.fit(Xtr_all, ytr_all)
y_pred_rf_all = rf_all.predict(Xte_all)

print("\n=== Baseline (ALL features, FULL dataset) ===")
print("GB F1:", f1_score(yte_all, y_pred_gb_all))
print("GB ROC-AUC:", roc_auc_score(yte_all, y_prob_gb_all))
print("RF F1:", f1_score(yte_all, y_pred_rf_all))


# ---------- 4) 只用 Top-K 特征重新训练 ----------
X_top = X_full_df[top_features].values
Xtr_top, Xte_top, ytr_top, yte_top = train_test_split(
    X_top, y_full, test_size=0.3, random_state=42, stratify=y_full
)

gb_top = train_gb_pipeline(Xtr_top, ytr_top)
y_pred_gb_top = gb_top.predict(Xte_top)
y_prob_gb_top = gb_top.predict_proba(Xte_top)[:, 1]

rf_top = RandomForest(n_estimators=5, max_depth=4,
                      sample_size=min(64, len(Xtr_top)), random_state=42)
rf_top.fit(Xtr_top, ytr_top)
y_pred_rf_top = rf_top.predict(Xte_top)

print("\n=== Feature-selected (TOP-K features, FULL dataset) ===")
print("GB F1:", f1_score(yte_top, y_pred_gb_top))
print("GB ROC-AUC:", roc_auc_score(yte_top, y_prob_gb_top))
print("RF F1:", f1_score(yte_top, y_pred_rf_top))
