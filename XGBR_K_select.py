import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# 1) Load data
df = pd.read_csv("/content/ea_b3lyp_opt_all_mol_all_features.csv")
df = df.drop(columns=["mol-no","error_ea","smiles"], errors="ignore")
df = df.dropna(subset=["g4mp2_ea"]).reset_index(drop=True)

y = df.pop("g4mp2_ea")
X = df.copy()

# 2) 60:20:20 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
# Now: train=0.6, val=0.2, test=0.2

# 3) build DMatrices for full-set (we'll remake inside loop after feature‐select)
dval_full = None  # placeholder

# parameters to try
param = {
    "objective":        "reg:squarederror",
    "eval_metric":      "mae",
    "eta":              0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "seed":             42,
}

# 4) loop over k
n_features = X.shape[1]
ks = list(range(10, n_features+1, 10))
if ks[-1] != n_features:
    ks.append(n_features)

best_mae = np.inf
best_k   = None
best_feats = None

for k in ks:
    # a) select top-k
    sel = SelectKBest(f_regression, k=k)
    sel.fit(X_train, y_train)
    feat_idx = sel.get_support()
    feats = X.columns[feat_idx]
    # transform
    XT = sel.transform(X_train)
    XV = sel.transform(X_val)
    XTe= sel.transform(X_test)

    # b) make DMatrix
    dtrain = xgb.DMatrix(XT, label=y_train)
    dval   = xgb.DMatrix(XV, label=y_val)
    dtest  = xgb.DMatrix(XTe, label=y_test)

    # c) train with early stopping
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain,"train"), (dval,"valid")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # d) predict & eval on test
    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
    mae   = mean_absolute_error(y_test, ypred)

    print(f"k={k:3d}  |  test MAE = {mae:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_k   = k
        best_feats = list(feats)

print("\n→ Best k :", best_k)
print("→ Best test MAE :", best_mae)
print("→ Selected features:")
for f in best_feats:
    print("   ", f)
