# student_project/student_project.py
"""
Student starter (broken by design) for CS5100 Phase 1.
This file is intentionally TODO-heavy. Students must implement the functions
below to pass the tests.

Design goals:
- Clear error messages (NotImplementedError) instead of silent wrong types.
- Helpful guidance in docstrings about expected behavior.
- Safe to import (no heavy compute at import time).
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import Counter

# -------------------------
# Section A: Data Loading
# -------------------------
def load_data(path=None):
    """
    Load the student dataset following the rules required by the autograder.
    This function MUST NOT modify the dataset.
    """

    if path is not None:
        # auto-detect separator: only check first line
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        return pd.read_csv(path, sep=sep)

    candidate_paths = [
        "student-mat-mini.csv",
        "datasets/student-mat-mini.csv",
        "datasets/student-mat.csv",
        "student-mat.csv",
    ]

    for p in candidate_paths:
        if Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                first_line = f.readline()
            sep = ";" if ";" in first_line else ","
            return pd.read_csv(p, sep=sep)

    raise FileNotFoundError("Could not find dataset file.")



# -------------------------
# Section B: Exploratory / Preprocessing helpers
# -------------------------
def summary_stats():
    """
    Return a dictionary with mean G3 and median absences.
    """
    df = load_data()
    return {
        "mean_G3": float(df["G3"].mean()),
        "median_absences": int(df["absences"].median())
    }



def compute_correlations():
    """
    Compute correlations for numeric columns only.
    """
    df = load_data()
    return df.corr(numeric_only=True)



def preprocess_data(df):
    """
    Preprocess the provided DataFrame and return a processed DataFrame ready for modeling.

    Expected contract (must meet autograder checks):
    - Create target column 'at_risk' as: (df['G3'] < 10).astype(int)
    - Drop grade columns (G1, G2, G3) from the feature matrix to avoid leakage
    - Encode categorical variables (one-hot or similar) so NO object dtypes remain
    - Impute missing values
    - Scale numeric columns to [0,1] range
    - Return a pandas DataFrame that includes 'at_risk' and only numeric columns otherwise
    """
    df = df.copy()

    # 1) 创建目标列 at_risk
    at_risk = (df["G3"] < 10).astype(int)

    # 2) 删除 G1, G2, G3（防止泄漏）
    feature_df = df.drop(columns=["G1", "G2", "G3"])

    # 3) 区分数值列和类别列
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    cat_cols = feature_df.columns.difference(numeric_cols)

    # 4) 缺失值填补
    # 数值列用中位数
    if len(numeric_cols) > 0:
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(
            feature_df[numeric_cols].median()
        )
    # 类别列用众数
    if len(cat_cols) > 0:
        mode_vals = feature_df[cat_cols].mode().iloc[0]
        feature_df[cat_cols] = feature_df[cat_cols].fillna(mode_vals)

    # 5) 对类别变量做 one-hot 编码
    feature_df = pd.get_dummies(feature_df, columns=list(cat_cols), drop_first=False)

    # 现在 feature_df 全是数值列
    feature_df = feature_df.astype(float)

    # 6) 将所有特征列缩放到 [0,1]
    min_vals = feature_df.min()
    max_vals = feature_df.max()
    denom = (max_vals - min_vals).replace(0, 1)  # 防止除以 0
    feature_df = (feature_df - min_vals) / denom

    # 7) 最后再加上 at_risk，不参与缩放，保持 0/1
    feature_df["at_risk"] = at_risk.values.astype(int)

    return feature_df



# -------------------------
# Section B: Gradient Boosting Pipeline (Broken starter)
# -------------------------
def train_gb_pipeline(X_train=None, y_train=None):
    """
    Build and fit a sklearn Pipeline that includes:
      ("preprocessor", ColumnTransformer(...)) and ("classifier", GradientBoostingClassifier)

    - Must return a fitted sklearn-like pipeline with .predict() and preferably .predict_proba()
    - Tests expect a named step "preprocessor" to exist (if you return a Pipeline)
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    except Exception:
        raise NotImplementedError(
            "sklearn not available in the environment; install dependencies."
        )

    # 如果没传 X_train / y_train，就自己从原始数据构建
    if X_train is None or y_train is None:
        df = load_data()
        processed = preprocess_data(df)
        y = processed["at_risk"].values
        X = processed.drop(columns=["at_risk"])
    else:
        X = X_train
        y = y_train

    # 如果是 DataFrame，就按列名构建一个 ColumnTransformer
    if isinstance(X, pd.DataFrame):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        transformers = []
        if numeric_features:
            # 数值列：这里再做一次 [0,1] 缩放（即使已经缩放过也没关系）
            transformers.append(("num", MinMaxScaler(), numeric_features))
        if categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            )

        if transformers:
            preprocessor = ColumnTransformer(transformers)
        else:
            # 全是数值特征时，就直接 passthrough
            preprocessor = "passthrough"
    else:
        # 如果传进来的是 numpy 数组，直接 passthrough
        preprocessor = "passthrough"

    gb = GradientBoostingClassifier(random_state=42)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # 重要：名字必须叫 "preprocessor"
            ("classifier", gb),
        ]
    )

    model.fit(X, y)
    return model



# -------------------------
# Section C: Random Forest (From Scratch) skeleton
# -------------------------
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        counts = Counter(y)
        impurity = 1.0
        for c in counts.values():
            p = c / len(y)
            impurity -= p**2
        return impurity

    def _best_split(self, X, y):
        best_feature, best_thresh, best_gain = None, None, 0.0
        current_impurity = self._gini(y)
        n_features = X.shape[1]

        for feature in range(n_features):
            values = X[:, feature]
            thresholds = np.unique(values)
            for t in thresholds:
                left_idx = values < t
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                left_y = y[left_idx]
                right_y = y[right_idx]

                p = len(left_y) / len(y)
                gain = current_impurity - (p * self._gini(left_y) + (1 - p) * self._gini(right_y))

                if gain > best_gain:
                    best_feature, best_thresh, best_gain = feature, t, gain

        return best_feature, best_thresh

    def _build_tree(self, X, y, depth):
        # If pure or max depth reached → leaf
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return {"leaf": True, "prediction": Counter(y).most_common(1)[0][0]}

        feature, thresh = self._best_split(X, y)
        if feature is None:
            return {"leaf": True, "prediction": Counter(y).most_common(1)[0][0]}

        left_idx = X[:, feature] < thresh
        right_idx = ~left_idx

        return {
            "leaf": False,
            "feature": feature,
            "threshold": thresh,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1),
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, node, x):
        if node["leaf"]:
            return node["prediction"]
        if x[node["feature"]] < node["threshold"]:
            return self._predict_one(node["left"], x)
        else:
            return self._predict_one(node["right"], x)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(self.tree, x) for x in X])



class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, sample_size=None, random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            size = self.sample_size if self.sample_size is not None else n
            idx = rng.choice(n, size=size, replace=True)

            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        # predictions shape = (n_estimators, n_samples)
        preds = np.array([tree.predict(X) for tree in self.trees])

        final = []
        for col in preds.T:  # majority vote per sample
            final.append(Counter(col).most_common(1)[0][0])

        return np.array(final)

