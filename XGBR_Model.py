!pip install xgboost --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, callback as xgb_callback # Import callback module
from sklearn.feature_selection import SelectKBest, f_regression

# 1) Load & prep data
df = pd.read_csv('features.csv')
df = df.dropna(subset=['g4mp2_ea']).reset_index(drop=True)

# Split out X/y
y = df.pop('g4mp2_ea')
X = df.copy()

print("Input dataset shape:", X.shape)

# 2) Optional: model‐based feature selection
#    here we pick the top K by univariate f_regression
#    you can tune K as you like
selector = SelectKBest(score_func=f_regression, k=50)
X_selected = selector.fit_transform(X, y)
selected_cols = X.columns[selector.get_support()].tolist()
print("Selected top features:", selected_cols)
X = pd.DataFrame(X_selected, columns=selected_cols)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4) Set up XGB + randomized search
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators':    [100, 300, 500, 800, 1200],
    'learning_rate':   [0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth':       [3, 5, 7, 9, 12],
    'subsample':       [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha':       [0, 0.1, 0.5, 1, 5],
    'reg_lambda':      [1, 5, 10, 20],
    'min_child_weight':[1, 3, 5, 7]
}

search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=60,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best XGB params:", search.best_params_)

# 5) Evaluate on train & test
y_pred_train = best_model.predict(X_train)
y_pred_test  = best_model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_test  = mean_absolute_error(y_test,  y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test,  y_pred_test))

print(f"\nTrain MAE:  {mae_train:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test MAE:   {mae_test:.3f}")
print(f"Test RMSE:  {rmse_test:.3f}")

# 6) Plotting

# a) first selected feature vs actual & predicted
feat0 = selected_cols[0]
plt.figure(figsize=(6,6))
plt.scatter(X_test[feat0], y_test,  c='red',   label='Actual')
plt.scatter(X_test[feat0], y_pred_test, c='blue',  label='Predicted')
# best‐fit line
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression().fit(
    X_test[[feat0]], y_pred_test
)
plt.plot(
    X_test[feat0],
    lr1.predict(X_test[[feat0]]),
    c='green', lw=2, label='Fit: Pred vs '+feat0
)
plt.xlabel(feat0)
plt.ylabel('g4mp2_ea')
plt.title('First Feature vs Actual & Predicted EA')
plt.legend()
plt.tight_layout()
plt.show()

# b) actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, c='blue', alpha=0.7, label='Predictions')
lr2 = LinearRegression().fit(
    y_test.values.reshape(-1,1), y_pred_test
)
y_line = lr2.predict(y_test.values.reshape(-1,1))
plt.plot(y_test, y_line,     c='green', lw=2, label='Fit: Pred vs Actual')
m, M = y_test.min(), y_test.max()
plt.plot([m,M],[m,M], '--', c='red',   lw=2, label='Ideal y=x')
plt.xlabel('Actual g4mp2_ea')
plt.ylabel('Predicted g4mp2_ea')
plt.title('Actual vs Predicted EA')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
