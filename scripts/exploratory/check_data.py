import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/intermediate/task_imputed_matrix.csv')
X_df = df.select_dtypes(include=[np.number])
print(X_df.columns[:5])
print(X_df.max().max())
print(X_df.min().min())
