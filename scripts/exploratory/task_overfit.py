import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('data/processed/intermediate/task_imputed_matrix.csv')
X_df = df.select_dtypes(include=[np.number])

bad_cols = X_df.columns[(X_df.max() > 2) | (X_df.min() < -1)]
good_cols = [c for c in X_df.columns if c not in bad_cols]
X = X_df[good_cols].values

variances = X.var(axis=0)
mask = variances > 0
X_filtered = X[:, mask]

eps = 0.005
X_smoothed = np.clip(X_filtered, eps, 1 - eps)
X_logit = np.log(X_smoothed / (1 - X_smoothed))

# Hold out system 0
train_X = X_logit[1:]
test_X = X_logit[0:1]

# Try to predict task 0
target_task = 0
y_train = train_X[:, target_task]

# Find correlations with all other tasks
corrs = []
for i in range(train_X.shape[1]):
    if i == target_task:
        corrs.append(-1)
        continue
    # Pearson correlation
    c = np.corrcoef(y_train, train_X[:, i])[0, 1]
    corrs.append(abs(c) if not np.isnan(c) else -1)

corrs = np.array(corrs)
best_5 = np.argsort(corrs)[-5:]
print(f"Top 5 training correlations: {corrs[best_5]}")

# Train OLS
X_train_features = train_X[:, best_5]
model = LinearRegression().fit(X_train_features, y_train)
train_r2 = model.score(X_train_features, y_train)
print(f"Train R^2: {train_r2:.4f}")

# Test
X_test_features = test_X[:, best_5]
pred = model.predict(X_test_features)
actual = test_X[0, target_task]
print(f"Predicted logit: {pred[0]:.2f}, Actual logit: {actual:.2f}")

# Convert back to prob
pred_prob = 1 / (1 + np.exp(-pred[0]))
actual_prob = 1 / (1 + np.exp(-actual))
print(f"Predicted prob: {pred_prob:.2f}, Actual prob: {actual_prob:.2f}")

