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
print("\n--- PCA on LOGIT scores (Model Only, n=8) ---")
for i, ev in enumerate(pca_model.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

pca_full = PCA()
pca_full.fit(X_logit)
print("\n--- PCA on LOGIT scores (All 16 systems) ---")
for i, ev in enumerate(pca_full.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {ev*100:.1f}%")

# Let's check PC2 scores for all 16 systems to see if it separates agents
pc_scores = pca_full.transform(X_logit)
print("\n--- PC2 scores ---")
for i, (model, agent) in enumerate(zip(df['model'], df['agent'])):
    print(f"{model} + {agent}: PC2 = {pc_scores[i, 1]:.2f}")

