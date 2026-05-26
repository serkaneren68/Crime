"""
CSE 555 Final Project - Communities and Crime Analysis
Author: Serkan Eren
"""
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from collections import Counter
from minisom import MiniSom

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})

# ----------------------------------------------------------------------------
# 0. Load dataset
# ----------------------------------------------------------------------------
attr_pattern = re.compile(r"^@attribute\s+(\S+)\s+\S+", re.IGNORECASE)
names_path = os.path.join(ROOT, "communities.names")
data_path = os.path.join(ROOT, "communities.data")

attr_names = []
with open(names_path) as fh:
    for line in fh:
        m = attr_pattern.match(line.strip())
        if m:
            attr_names.append(m.group(1))

df_full = pd.read_csv(data_path, header=None, names=attr_names, na_values="?")
print(f"Loaded dataset: {df_full.shape}, attributes: {len(attr_names)}")

# Drop non-predictive identifiers
non_predictive = ["state", "county", "community", "communityname", "fold"]
df_full = df_full.drop(columns=non_predictive)

TARGET = "ViolentCrimesPerPop"

# We pick a curated subset (>10 features) of well-known sociodemographic features
# with low missingness. All are already normalized to [0,1].
selected = [
    "population", "householdsize", "racepctblack", "racePctWhite",
    "agePct12t29", "agePct65up", "pctUrban", "medIncome",
    "PctPopUnderPov", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
    "MalePctDivorce", "PctKids2Par", "PctIlleg", "NumImmig",
    "PctHousOwnOcc", "PctVacantBoarded",
]
missing = df_full[selected + [TARGET]].isna().sum()
print("Missing values per selected feature:\n", missing)

df = df_full[selected + [TARGET]].copy()
# Median imputation for any remaining missing values
df[selected] = df[selected].fillna(df[selected].median())
df = df.dropna(subset=[TARGET])

# ----------------------------------------------------------------------------
# Discretize target into 3 classes by tertiles -> Low / Medium / High
# ----------------------------------------------------------------------------
labels = ["Low", "Medium", "High"]
df["CrimeClass"] = pd.qcut(df[TARGET], q=3, labels=labels)
class_counts = df["CrimeClass"].value_counts().sort_index()
print("\nClass distribution:\n", class_counts)

X = df[selected].values
y = df["CrimeClass"].astype(str).values
classes = labels
class_colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
color_arr = np.array([class_colors[c] for c in y])

# Save a small summary used in the report
summary = {
    "n_samples": int(df.shape[0]),
    "n_features": len(selected),
    "feature_names": selected,
    "class_counts": {str(k): int(v) for k, v in class_counts.items()},
}
with open(os.path.join(ROOT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

# ----------------------------------------------------------------------------
# Task 1 - Boxplots + outlier detection (IQR method)
# ----------------------------------------------------------------------------
print("\n[Task 1] Boxplots and outlier detection ...")
fig, ax = plt.subplots(figsize=(13, 6))
df[selected].boxplot(ax=ax, rot=75, grid=False)
ax.set_title("Feature distributions (before outlier removal)")
ax.set_ylabel("Normalized value")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "boxplot_before.png"))
plt.close(fig)

# Multivariate outlier detection via Isolation Forest. A multivariate
# detector is preferred over per-feature IQR masking because the latter
# discards rows that are extreme on any single dimension -- this would
# remove ~18% of the data and severely unbalance the class distribution
# (the High-crime class is intrinsically extreme on several axes). We use
# IsolationForest with a modest contamination=0.05 so that only the most
# globally anomalous communities are removed.
iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
iso_pred = iso.fit_predict(df[selected].values)  # +1 inlier, -1 outlier
inside = iso_pred == 1
n_removed = int((~inside).sum())
print(f"IsolationForest removed {n_removed} rows ({n_removed/len(df)*100:.1f}%)")

df_clean = df[inside].reset_index(drop=True)
print(f"Cleaned dataset shape: {df_clean.shape}")
print("Class distribution after cleaning:\n", df_clean["CrimeClass"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(13, 6))
df_clean[selected].boxplot(ax=ax, rot=75, grid=False)
ax.set_title("Feature distributions after IsolationForest cleaning (5%)")
ax.set_ylabel("Normalized value")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "boxplot_after.png"))
plt.close(fig)

# Use cleaned dataset for the rest of the analysis
X = df_clean[selected].values
y = df_clean["CrimeClass"].astype(str).values
classes = labels
color_arr = np.array([class_colors[c] for c in y])

print("\nDone Task 1.")

# ----------------------------------------------------------------------------
# Task 2 - Correlation analysis
# ----------------------------------------------------------------------------
print("\n[Task 2] Correlation analysis ...")
# Encode class as ordinal for correlation with class
class_to_int = {"Low": 0, "Medium": 1, "High": 2}
y_int = np.array([class_to_int[c] for c in y])
corr_df = pd.DataFrame(X, columns=selected)
corr_df["CrimeClass"] = y_int
corr_matrix = corr_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Pearson correlation matrix (features + ordinal class)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "correlation_heatmap.png"))
plt.close(fig)

class_corr = corr_matrix["CrimeClass"].drop("CrimeClass").sort_values(key=lambda s: s.abs(), ascending=False)
print("Top |correlations| with class (Pearson):\n", class_corr.head(10))

# Spearman is more appropriate for an ordinal class label; report both.
spearman_class = pd.Series(
    {f: stats.spearmanr(X[:, i], y_int).correlation for i, f in enumerate(selected)}
).sort_values(key=lambda s: s.abs(), ascending=False)
print("\nTop |correlations| with class (Spearman):\n", spearman_class.head(10))
# Sanity: rank agreement between Pearson and Spearman feature-class correlations
rank_agree = stats.spearmanr(class_corr.abs(), spearman_class.reindex(class_corr.index).abs()).correlation
print(f"Pearson vs Spearman |corr| rank agreement (Spearman): {rank_agree:.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
class_corr.plot(kind="barh", ax=ax,
                color=["#d62728" if v > 0 else "#1f77b4" for v in class_corr.values])
ax.invert_yaxis()
ax.set_xlabel("Pearson correlation with crime class (ordinal)")
ax.set_title("Per-feature correlation with class label")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "feature_class_correlation.png"))
plt.close(fig)
print("Done Task 2.")

# ----------------------------------------------------------------------------
# Task 3 - Z-score normalization + Fisher Distance per feature
# ----------------------------------------------------------------------------
print("\n[Task 3] Z-score + Fisher Distance per feature ...")
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

def fisher_distance_per_feature(Xmat, labels_arr):
    """Multi-class Fisher score: J = trace(S_B) / trace(S_W) per feature."""
    classes_local = np.unique(labels_arr)
    overall_mean = Xmat.mean(axis=0)
    sb = np.zeros(Xmat.shape[1])
    sw = np.zeros(Xmat.shape[1])
    for c in classes_local:
        Xc = Xmat[labels_arr == c]
        nc = Xc.shape[0]
        mu_c = Xc.mean(axis=0)
        sb += nc * (mu_c - overall_mean) ** 2
        sw += ((Xc - mu_c) ** 2).sum(axis=0)
    return sb / np.where(sw == 0, 1e-12, sw)

fisher_raw = fisher_distance_per_feature(Xz, y)
fisher_series = pd.Series(fisher_raw, index=selected).sort_values(ascending=False)
print("Fisher Distance (z-scored features):\n", fisher_series)

fig, ax = plt.subplots(figsize=(9, 5))
fisher_series.plot(kind="barh", ax=ax, color="#3a7bb0")
ax.invert_yaxis()
ax.set_xlabel("Fisher Distance  (trace(S_B)/trace(S_W))")
ax.set_title("Per-feature Fisher Distance (z-scored features)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fisher_features.png"))
plt.close(fig)
print("Done Task 3.")

# ----------------------------------------------------------------------------
# Task 4 - PCA + Fisher Distance per PC
# ----------------------------------------------------------------------------
print("\n[Task 4] PCA and Fisher Distance per PC ...")
pca = PCA(n_components=Xz.shape[1])
Xpca = pca.fit_transform(Xz)
eigvals = pca.explained_variance_
expl_ratio = pca.explained_variance_ratio_

fisher_pcs = fisher_distance_per_feature(Xpca, y)
pc_names = [f"PC{i+1}" for i in range(Xz.shape[1])]
pc_df = pd.DataFrame({"eigenvalue": eigvals,
                      "explained_var_ratio": expl_ratio,
                      "fisher": fisher_pcs}, index=pc_names)
print(pc_df)

corr_eig_fisher = np.corrcoef(eigvals, fisher_pcs)[0, 1]
print(f"Pearson corr(eigenvalue, Fisher of PC) = {corr_eig_fisher:.3f}")
spearman_eig_fisher = stats.spearmanr(eigvals, fisher_pcs).correlation
print(f"Spearman corr(eigenvalue, Fisher of PC) = {spearman_eig_fisher:.3f}")

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fisher_series.plot(kind="bar", ax=axes[0], color="#3a7bb0")
axes[0].set_title("Fisher Distance per original (z-scored) feature")
axes[0].set_ylabel("Fisher Distance")
axes[0].tick_params(axis="x", rotation=75)
pc_df["fisher"].plot(kind="bar", ax=axes[1], color="#c0504d")
axes[1].set_title("Fisher Distance per principal component")
axes[1].set_ylabel("Fisher Distance")
axes[1].tick_params(axis="x", rotation=75)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fisher_comparison.png"))
plt.close(fig)

# Eigenvalue vs Fisher scatter
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(eigvals, fisher_pcs, color="#444444")
for i, name in enumerate(pc_names):
    ax.annotate(name, (eigvals[i], fisher_pcs[i]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Eigenvalue (variance of PC)")
ax.set_ylabel("Fisher Distance of PC")
ax.set_title(f"Eigenvalue vs Fisher Distance  (Pearson={corr_eig_fisher:.2f}, "
             f"Spearman={spearman_eig_fisher:.2f})")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "eigval_vs_fisher.png"))
plt.close(fig)

# Scree
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, len(expl_ratio) + 1), expl_ratio, color="#3a7bb0", alpha=0.8)
ax.plot(range(1, len(expl_ratio) + 1), np.cumsum(expl_ratio),
        color="#d62728", marker="o", label="cumulative")
ax.set_xlabel("Principal component index")
ax.set_ylabel("Explained variance ratio")
ax.set_title("PCA scree plot")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "pca_scree.png"))
plt.close(fig)
print("Done Task 4.")

# ----------------------------------------------------------------------------
# Task 5 - Scatter plots of top-2 / bottom-2 PCs
# ----------------------------------------------------------------------------
print("\n[Task 5] Scatter plots of PCs ...")
def scatter_pcs(idx1, idx2, fname, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in classes:
        mask = (y == c)
        ax.scatter(Xpca[mask, idx1], Xpca[mask, idx2],
                   s=14, alpha=0.65, label=c, color=class_colors[c],
                   edgecolors="none")
    ax.set_xlabel(f"PC{idx1+1}  (var={expl_ratio[idx1]*100:.1f}%)")
    ax.set_ylabel(f"PC{idx2+1}  (var={expl_ratio[idx2]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(title="Crime class")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)

scatter_pcs(0, 1, "pca_top2.png", "Projection onto the two most important PCs")
n = Xz.shape[1]
scatter_pcs(n - 2, n - 1, "pca_bottom2.png", "Projection onto the two least important PCs")
print("Done Task 5.")

# ----------------------------------------------------------------------------
# Task 6 - LDA and class-separability metric
# ----------------------------------------------------------------------------
print("\n[Task 6] LDA + separability metrics ...")
lda = LinearDiscriminantAnalysis(n_components=2)
Xlda = lda.fit_transform(Xz, y)

fig, ax = plt.subplots(figsize=(7, 5.5))
for c in classes:
    mask = (y == c)
    ax.scatter(Xlda[mask, 0], Xlda[mask, 1], s=14, alpha=0.7,
               label=c, color=class_colors[c], edgecolors="none")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_title("LDA 2D projection")
ax.legend(title="Crime class")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "lda_scatter.png"))
plt.close(fig)

# Quantitative separability: silhouette and KNN cross-val accuracy on the 2D
def class_silhouette(Xmat, labels_arr):
    return silhouette_score(Xmat, labels_arr)

def knn_cv_accuracy(Xmat, labels_arr, k=5, folds=5):
    clf = KNeighborsClassifier(n_neighbors=k)
    return cross_val_score(clf, Xmat, labels_arr, cv=folds, scoring="accuracy").mean()

sep = {
    "PCA-2D silhouette": class_silhouette(Xpca[:, :2], y),
    "LDA-2D silhouette": class_silhouette(Xlda, y),
    "PCA-2D 5NN accuracy (5-fold)": knn_cv_accuracy(Xpca[:, :2], y),
    "LDA-2D 5NN accuracy (5-fold)": knn_cv_accuracy(Xlda, y),
}
print("Separability metrics:")
for k, v in sep.items():
    print(f"  {k}: {v:.3f}")
print("Done Task 6.")

# ----------------------------------------------------------------------------
# Task 7 - Clustering: K-Means, DBSCAN on PCA(2); t-SNE and SOM on original
# ----------------------------------------------------------------------------
print("\n[Task 7] Clustering ...")
X2 = Xpca[:, :2]

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X2)

db = DBSCAN(eps=0.5, min_samples=10)
db_labels = db.fit_predict(X2)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_db_noise = int((db_labels == -1).sum())
print(f"DBSCAN: {n_db_clusters} clusters, {n_db_noise} noise points")

tsne = TSNE(n_components=2, perplexity=30, init="pca",
            learning_rate="auto", random_state=42)
Xtsne = tsne.fit_transform(Xz)

# SOM on original (z-scored) data
som_size = 12
som = MiniSom(som_size, som_size, Xz.shape[1], sigma=1.5,
              learning_rate=0.5, random_seed=42)
som.random_weights_init(Xz)
som.train_random(Xz, 5000, verbose=False)
som_coords = np.array([som.winner(x) for x in Xz])  # (N,2)
# Add jitter so overlapping points are visible (for the scatter view)
jitter = np.random.normal(0, 0.25, size=som_coords.shape)
som_xy = som_coords + jitter

# Per-cell dominant class and U-matrix-style summaries for the SOM hit-map view
som_winner_class = {}
som_hit_count = np.zeros((som_size, som_size), dtype=int)
for (i, j), cls in zip(map(tuple, som_coords), y):
    som_winner_class.setdefault((i, j), []).append(cls)
    som_hit_count[i, j] += 1

dominant = np.full((som_size, som_size), -1)
purity = np.zeros((som_size, som_size))
for (i, j), cls_list in som_winner_class.items():
    c = Counter(cls_list)
    top_cls, top_n = c.most_common(1)[0]
    dominant[i, j] = labels.index(top_cls)
    purity[i, j] = top_n / len(cls_list)

def plot_clusters(Xmat, cluster_labels, true_labels, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for c in classes:
        mask = (true_labels == c)
        axes[0].scatter(Xmat[mask, 0], Xmat[mask, 1], s=14, alpha=0.7,
                        label=c, color=class_colors[c], edgecolors="none")
    axes[0].legend(title="True class")
    axes[0].set_title(title + " - true classes")
    cl_unique = np.unique(cluster_labels)
    palette = sns.color_palette("tab10", n_colors=max(3, len(cl_unique)))
    for i, cl in enumerate(cl_unique):
        mask = (cluster_labels == cl)
        lbl = "noise" if cl == -1 else f"cluster {cl}"
        col = "#999999" if cl == -1 else palette[i % len(palette)]
        axes[1].scatter(Xmat[mask, 0], Xmat[mask, 1], s=14, alpha=0.7,
                        label=lbl, color=col, edgecolors="none")
    axes[1].legend(title="Cluster", fontsize=8)
    axes[1].set_title(title + " - clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)

plot_clusters(X2, km_labels, y, "cluster_kmeans.png", "K-Means on PCA(2)")
plot_clusters(X2, db_labels, y, "cluster_dbscan.png", "DBSCAN on PCA(2)")
# For t-SNE / SOM, plot true classes only (no clustering algorithm applied
# beyond the embedding itself, per the assignment wording)
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in classes:
    mask = (y == c)
    ax.scatter(Xtsne[mask, 0], Xtsne[mask, 1], s=14, alpha=0.7,
               label=c, color=class_colors[c], edgecolors="none")
ax.set_title("t-SNE embedding of original data (true classes)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(title="Crime class")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cluster_tsne.png"))
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Panel 1: scatter of winners coloured by true class (jittered for visibility)
for c in classes:
    mask = (y == c)
    axes[0].scatter(som_xy[mask, 0], som_xy[mask, 1], s=14, alpha=0.7,
                    label=c, color=class_colors[c], edgecolors="none")
axes[0].set_title("SOM winners (true class)")
axes[0].set_xlabel("SOM x"); axes[0].set_ylabel("SOM y")
axes[0].legend(title="Crime class", fontsize=8)

# Panel 2: per-cell hit count (how many samples landed on each neuron)
im2 = axes[1].imshow(som_hit_count.T, origin="lower", cmap="viridis", aspect="equal")
axes[1].set_title("SOM hit count per neuron")
axes[1].set_xlabel("SOM x"); axes[1].set_ylabel("SOM y")
fig.colorbar(im2, ax=axes[1], fraction=0.046)

# Panel 3: dominant class per cell, with purity as alpha
from matplotlib.colors import ListedColormap
cmap_dom = ListedColormap(["#f0f0f0"] + [class_colors[c] for c in labels])  # -1 -> light grey
display_grid = dominant + 1  # shift -1 (empty) to 0
axes[2].imshow(display_grid.T, origin="lower", cmap=cmap_dom,
               vmin=0, vmax=len(labels), aspect="equal", alpha=1.0)
# overlay purity as text in non-empty cells
for i in range(som_size):
    for j in range(som_size):
        if dominant[i, j] >= 0:
            axes[2].text(i, j, f"{int(purity[i,j]*100)}", ha="center",
                         va="center", fontsize=6, color="white")
axes[2].set_title("Dominant class per neuron (% purity)")
axes[2].set_xlabel("SOM x"); axes[2].set_ylabel("SOM y")
# legend patches
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=class_colors[c], label=c) for c in labels]
handles.append(mpatches.Patch(color="#f0f0f0", label="empty"))
axes[2].legend(handles=handles, fontsize=7, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "cluster_som.png"))
plt.close(fig)

# Quantify how class-organised the SOM is
nonempty = dominant.flatten() >= 0
mean_purity = float(purity.flatten()[nonempty].mean())
print(f"SOM mean cell purity (non-empty cells): {mean_purity:.3f}")

# Cluster vs class agreement
cluster_metrics = {
    "KMeans ARI": adjusted_rand_score(y_int, km_labels),
    "KMeans NMI": normalized_mutual_info_score(y_int, km_labels),
    "DBSCAN ARI": adjusted_rand_score(y_int, db_labels),
    "DBSCAN NMI": normalized_mutual_info_score(y_int, db_labels),
}
print("Cluster agreement:")
for k, v in cluster_metrics.items():
    print(f"  {k}: {v:.3f}")
print("Done Task 7.")

# ----------------------------------------------------------------------------
# Task 8 - Hypothesis test on PctKids2Par mean (Low vs High classes)
# ----------------------------------------------------------------------------
print("\n[Task 8] Hypothesis test ...")
feature_for_test = "PctKids2Par"
g_low = df_clean.loc[df_clean["CrimeClass"] == "Low", feature_for_test].values
g_high = df_clean.loc[df_clean["CrimeClass"] == "High", feature_for_test].values

ttest_results = {}
for n in (36, 64):
    rng = np.random.default_rng(42)
    s_low = rng.choice(g_low, size=n, replace=False)
    s_high = rng.choice(g_high, size=n, replace=False)
    t, p = stats.ttest_ind(s_low, s_high, equal_var=False)
    ttest_results[n] = {"t": float(t), "p": float(p),
                        "mean_low": float(s_low.mean()),
                        "mean_high": float(s_high.mean()),
                        "std_low": float(s_low.std(ddof=1)),
                        "std_high": float(s_high.std(ddof=1))}
    print(f"  n={n}: t={t:.3f}, p={p:.3e}, mean_low={s_low.mean():.3f}, mean_high={s_high.mean():.3f}")

# Population (full-class) reference
t_full, p_full = stats.ttest_ind(g_low, g_high, equal_var=False)
print(f"  population: t={t_full:.3f}, p={p_full:.3e}")
print("Done Task 8.")

# ----------------------------------------------------------------------------
# Save metrics for the report
# ----------------------------------------------------------------------------
results = {
    "summary": summary,
    "outlier_removed": int(n_removed),
    "class_counts_clean": {k: int(v) for k, v in df_clean["CrimeClass"].value_counts().sort_index().items()},
    "fisher_features": fisher_series.to_dict(),
    "fisher_pcs": pc_df.to_dict(orient="index"),
    "eigval_fisher_pearson": float(corr_eig_fisher),
    "eigval_fisher_spearman": float(spearman_eig_fisher),
    "separability": {k: float(v) for k, v in sep.items()},
    "clustering": {k: float(v) for k, v in cluster_metrics.items()},
    "dbscan_clusters": int(n_db_clusters),
    "dbscan_noise": int(n_db_noise),
    "ttest": ttest_results,
    "ttest_population": {"t": float(t_full), "p": float(p_full)},
    "feature_for_test": feature_for_test,
    "class_corr_top_pearson": class_corr.head(10).to_dict(),
    "class_corr_top_spearman": spearman_class.head(10).to_dict(),
    "pearson_spearman_rank_agreement": float(rank_agree),
    "som_mean_cell_purity": mean_purity,
}
with open(os.path.join(ROOT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("\nAll tasks complete. Figures in:", FIG)

