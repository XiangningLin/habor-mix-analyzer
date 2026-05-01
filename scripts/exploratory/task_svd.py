import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('data/processed/intermediate/task_imputed_matrix.csv')
X_df = df.select_dtypes(include=[np.number])
# Drop Unnamed if exists
if 'Unnamed: 0' in X_df.columns:
    X_df = X_df.drop(columns=['Unnamed: 0'])

X = X_df.values

variances = X.var(axis=0)
mask = variances > 0
X_filtered = X[:, mask]

print(f"Original shape: {X.shape}")
print(f"Filtered shape: {X_filtered.shape} (removed {(~mask).sum()} zero-variance tasks)")

pca_raw = PCA()
pca_raw.fit(X_filtered)
print("\n--- PCA on RAW scores (16 systems) ---")
for i, ev in enumerate(pca_raw.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

eps = 0.005
X_smoothed = np.clip(X_filtered, eps, 1 - eps)
X_logit = np.log(X_smoothed / (1 - X_smoothed))

pca_logit = PCA()
pca_logit.fit(X_logit)
print("\n--- PCA on LOGIT scores (16 systems) ---")
for i, ev in enumerate(pca_logit.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

is_model_only = df['agent'] == 'terminus-2'
X_model_only = X_filtered[is_model_only]
pca_model = PCA()
pca_model.fit(X_model_only)
print("\n--- PCA on RAW scores (Model Only, n=8) ---")
for i, ev in enumerate(pca_model.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

pca_model_logit = PCA()
pca_model_logit.fit(X_logit[is_model_only])
print("\n--- PCA on LOGIT scores (Model Only, n=8) ---")
for i, ev in enumerate(pca_model_logit.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

