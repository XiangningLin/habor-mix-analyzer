import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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

is_model_only = df['agent'] == 'terminus-2'

pca_model = PCA()
pca_model.fit(X_logit[is_model_only])
model_pc1 = pca_model.explained_variance_ratio_[0] * 100

pca_full = PCA()
pca_full.fit(X_logit)
full_pc1 = pca_full.explained_variance_ratio_[0] * 100
full_pc2 = pca_full.explained_variance_ratio_[1] * 100

print(f"Task Model PC1: {model_pc1:.1f}%")
print(f"Task Full PC1: {full_pc1:.1f}%")
print(f"Task Full PC2: {full_pc2:.1f}%")

pc_scores = pca_full.transform(X_logit)
agent_pc2 = pc_scores[~is_model_only, 1]
model_pc2 = pc_scores[is_model_only, 1]

print(f"Agent PC2 mean: {agent_pc2.mean():.1f}")
print(f"Model PC2 mean: {model_pc2.mean():.1f}")
print(f"Agent PC2 min: {agent_pc2.min():.1f}")
print(f"Model PC2 max: {model_pc2.max():.1f}")

