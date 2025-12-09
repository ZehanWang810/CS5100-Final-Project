from student_project import load_data, preprocess_data, train_gb_pipeline, RandomForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


# ---------- 1) 载入 FULL 数据并预处理 ----------
df_full = load_data("datasets/student-mat.csv")
proc_full = preprocess_data(df_full)

X = proc_full.drop(columns=["at_risk"]).values
y = proc_full["at_risk"].values

print("Full dataset shape:", X.shape)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- 2) 单模型 Baseline（GB + RF） ----------
gb_base = train_gb_pipeline(X_tr, y_tr)
y_pred_gb = gb_base.predict(X_te)
y_prob_gb = gb_base.predict_proba(X_te)[:, 1]

rf_base = RandomForest(
    n_estimators=5, max_depth=4, sample_size=min(64, len(X_tr)), random_state=42
)
rf_base.fit(X_tr, y_tr)
y_pred_rf = rf_base.predict(X_te)

print("\n=== Baseline models on FULL dataset ===")
print("GB F1:", f1_score(y_te, y_pred_gb))
print("GB ROC-AUC:", roc_auc_score(y_te, y_prob_gb))
print("RF F1:", f1_score(y_te, y_pred_rf))


# ---------- 3) Stacking：用 KFold 做 out-of-fold meta-features ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

meta_train = np.zeros((len(X_tr), 2))  # [GB_prob, RF_pred]

for fold, (idx_tr, idx_val) in enumerate(kf.split(X_tr, y_tr), start=1):
    X_tr_f, X_val_f = X_tr[idx_tr], X_tr[idx_val]
    y_tr_f, y_val_f = y_tr[idx_tr], y_tr[idx_val]

    # 这一折上的 GB
    gb = train_gb_pipeline(X_tr_f, y_tr_f)
    gb_val_prob = gb.predict_proba(X_val_f)[:, 1]

    # 这一折上的 RF
    rf = RandomForest(
        n_estimators=5,
        max_depth=4,
        sample_size=min(64, len(X_tr_f)),
        random_state=42 + fold,  # 不同折换个随机种子
    )
    rf.fit(X_tr_f, y_tr_f)
    rf_val_pred = rf.predict(X_val_f)

    meta_train[idx_val, 0] = gb_val_prob
    meta_train[idx_val, 1] = rf_val_pred

# ---------- 4) 在整个训练集上重新训练 base models，用来生成测试集的 meta-features ----------
gb_full = train_gb_pipeline(X_tr, y_tr)
rf_full = RandomForest(
    n_estimators=5, max_depth=4, sample_size=min(64, len(X_tr)), random_state=99
)
rf_full.fit(X_tr, y_tr)

gb_test_prob = gb_full.predict_proba(X_te)[:, 1]
rf_test_pred = rf_full.predict(X_te)

meta_test = np.column_stack([gb_test_prob, rf_test_pred])

# ---------- 5) 训练 meta-learner（Logistic Regression） ----------
meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(meta_train, y_tr)

y_pred_stack = meta_clf.predict(meta_test)
y_prob_stack = meta_clf.predict_proba(meta_test)[:, 1]

print("\n=== Stacking (GB + RF -> LogisticRegression) ===")
print("Stacking F1:", f1_score(y_te, y_pred_stack))
print("Stacking ROC-AUC:", roc_auc_score(y_te, y_prob_stack))
