import pandas as pd
df = pd.read_csv('data/processed/intermediate/benchmark_observed_imputed_long.csv')
df = df.dropna(subset=["normalized_score"])
y = df["normalized_score"].astype(float)
print(f"Max y: {y.max()}")
print(f"Min y: {y.min()}")

# Find the rows with huge y
print(df.loc[y.abs() > 100, ['model', 'agent', 'benchmark', 'normalized_score']])
