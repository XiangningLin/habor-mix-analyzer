import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

df = pd.read_csv('data/processed/intermediate/benchmark_observed_imputed_long.csv')
df = df.dropna(subset=["normalized_score"])
y = df["normalized_score"].astype(float)

blocks = []
for term in ["agent", "benchmark"]:
    blocks.append(pd.get_dummies(df[term].astype(str), prefix=term, drop_first=True, dtype=float))
x = pd.concat(blocks, axis=1)

print(f"Condition number: {np.linalg.cond(x.values)}")

model = Ridge(alpha=1.0)
model.fit(x, y)
preds = model.predict(x)

print("--- RIDGE ---")
print(f"Max pred: {preds.max()}")
print(f"Min pred: {preds.min()}")
print(f"Max coef: {model.coef_.max()}")
print(f"Min coef: {model.coef_.min()}")

