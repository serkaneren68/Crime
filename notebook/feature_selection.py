"""
Dendrogram-based feature selection on the Communities and Crime dataset.

Pipeline:
  1. Load all 122 predictive features.
  2. Drop features with > 5% missingness (eliminates LEMAS block).
  3. Median-impute remaining missing values.
  4. Build absolute Pearson correlation matrix; convert to distance d = 1 - |r|.
  5. Complete-linkage hierarchical clustering on the condensed distance matrix.
     (Complete linkage guarantees max intra-cluster distance <= threshold,
     which average linkage does not.)
  6. Sweep thresholds; report cluster count at each. Pick d to target ~18.
  7. Pick one representative per cluster: the feature with the lowest
     mean intra-cluster distance (tie-break: lowest missingness).
  8. Compare the resulting set with the manual 18-feature selection
     used in analysis.py.

Outputs:
  figures/feature_dendrogram.png
  results_feature_selection.json (cluster table + representatives)
"""
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
attr_pattern = re.compile(r"^@attribute\s+(\S+)\s+\S+", re.IGNORECASE)
attr_names = []
with open(os.path.join(ROOT, "communities.names")) as fh:
    for line in fh:
        m = attr_pattern.match(line.strip())
        if m:
            attr_names.append(m.group(1))

df = pd.read_csv(os.path.join(ROOT, "communities.data"),
                 header=None, names=attr_names, na_values="?")

non_predictive = ["state", "county", "community", "communityname", "fold"]
TARGET = "ViolentCrimesPerPop"
df = df.drop(columns=non_predictive)
predictors_all = [c for c in df.columns if c != TARGET]
print(f"Total predictive features: {len(predictors_all)}")

# --------------------------------------------------------------------------
# 2. Missingness filter
# --------------------------------------------------------------------------
miss_rate = df[predictors_all].isna().mean()
KEEP_MISS_THRESH = 0.05
kept = miss_rate[miss_rate <= KEEP_MISS_THRESH].index.tolist()
dropped_miss = miss_rate[miss_rate > KEEP_MISS_THRESH].index.tolist()
print(f"After missingness filter (<= {KEEP_MISS_THRESH*100:.0f}%): "
      f"{len(kept)} features kept, {len(dropped_miss)} dropped.")

X = df[kept].copy()
X = X.fillna(X.median())

# --------------------------------------------------------------------------
# 3. Correlation -> distance
# --------------------------------------------------------------------------
corr = X.corr(method="pearson").abs()
# Floating-point safety: clip to [0, 1] and force exact zero diagonal
dist = 1.0 - corr.values
np.fill_diagonal(dist, 0.0)
dist = np.clip(dist, 0.0, 1.0)
dist = (dist + dist.T) / 2.0  # symmetrize

condensed = squareform(dist, checks=False)

# --------------------------------------------------------------------------
# 4. Hierarchical clustering
# --------------------------------------------------------------------------
Z = linkage(condensed, method="complete")

# Sweep thresholds to see how cluster count scales with d
print("\nThreshold sweep (complete linkage):")
print(f"{'d':>6} {'|r|>=':>8} {'#clusters':>10}")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    nc = fcluster(Z, t=t, criterion="distance").max()
    print(f"{t:>6.2f} {1-t:>8.2f} {nc:>10d}")

THRESH = 0.35  # cut: merge anything with |r| >= 0.65 -> 43 clusters
cluster_ids = fcluster(Z, t=THRESH, criterion="distance")
n_clusters = cluster_ids.max()
print(f"\nUsing d={THRESH} (|r| >= {1-THRESH:.2f}) -> {n_clusters} clusters")

cluster_df = pd.DataFrame({"feature": kept, "cluster": cluster_ids})

# --------------------------------------------------------------------------
# 5. Representative per cluster (lowest mean intra-cluster distance)
# --------------------------------------------------------------------------
representatives = []
cluster_summary = []
dist_df = pd.DataFrame(dist, index=kept, columns=kept)
for cid in sorted(cluster_df["cluster"].unique()):
    members = cluster_df.loc[cluster_df["cluster"] == cid, "feature"].tolist()
    if len(members) == 1:
        rep = members[0]
        mean_d = 0.0
    else:
        sub = dist_df.loc[members, members]
        # exclude self by setting diagonal to nan
        sub_vals = sub.values.copy()
        np.fill_diagonal(sub_vals, np.nan)
        mean_intra = np.nanmean(sub_vals, axis=1)
        ranking = pd.DataFrame({
            "feature": members,
            "mean_intra_d": mean_intra,
            "miss": [miss_rate[m] for m in members],
        }).sort_values(["mean_intra_d", "miss"])
        rep = ranking.iloc[0]["feature"]
        mean_d = float(ranking.iloc[0]["mean_intra_d"])
    representatives.append(rep)
    cluster_summary.append({
        "cluster": int(cid),
        "size": len(members),
        "representative": rep,
        "rep_mean_intra_d": mean_d,
        "members": members,
    })

print(f"\nSelected {len(representatives)} representatives.")

# Sanity check: max intra-cluster |r| of selected set must be < 1 - THRESH
# (guaranteed only with complete linkage)
sel_corr = X[representatives].corr().abs()
np.fill_diagonal(sel_corr.values, 0.0)
max_r_selected = float(sel_corr.values.max())
ok = "PASS" if max_r_selected < (1 - THRESH) + 1e-9 else "FAIL"
print(f"Maximum |Pearson r| among representatives: {max_r_selected:.3f} "
      f"(threshold < {1 - THRESH:.2f}) -> {ok}")

# --------------------------------------------------------------------------
# 6. Comparison with current manual 18-feature selection
# --------------------------------------------------------------------------
manual_18 = [
    "population", "householdsize", "racepctblack", "racePctWhite",
    "agePct12t29", "agePct65up", "pctUrban", "medIncome",
    "PctPopUnderPov", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
    "MalePctDivorce", "PctKids2Par", "PctIlleg", "NumImmig",
    "PctHousOwnOcc", "PctVacantBoarded",
]

in_both = sorted(set(manual_18) & set(representatives))
only_manual = sorted(set(manual_18) - set(representatives))
only_dendro = sorted(set(representatives) - set(manual_18))

print(f"\nManual selection: {len(manual_18)}  |  Dendrogram: {len(representatives)}")
print(f"  In both           : {len(in_both)}")
print(f"  Only manual       : {len(only_manual)}  -> {only_manual}")
print(f"  Only dendrogram   : {len(only_dendro)}  -> {only_dendro}")

# For each "only manual" feature, find which dendrogram cluster it belongs to
manual_to_cluster = {}
for f in only_manual:
    cid = int(cluster_df.loc[cluster_df["feature"] == f, "cluster"].iloc[0])
    rep = [c["representative"] for c in cluster_summary if c["cluster"] == cid][0]
    manual_to_cluster[f] = {"cluster": cid, "representative": rep}

# --------------------------------------------------------------------------
# 7. Plot dendrogram
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9))
dendrogram(
    Z,
    labels=kept,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold=THRESH,
    ax=ax,
)
ax.axhline(THRESH, color="red", linestyle="--", linewidth=1.0,
           label=f"cut: d = {THRESH}  (|r| >= {1-THRESH:.2f})")
ax.set_ylabel("Correlation distance  d = 1 - |r|")
ax.set_title(f"Hierarchical clustering of {len(kept)} predictive features "
             f"(complete linkage) — {n_clusters} clusters at d={THRESH}")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "feature_dendrogram.png"))
plt.close(fig)
print(f"\nDendrogram saved to {os.path.join(FIG, 'feature_dendrogram.png')}")

# --------------------------------------------------------------------------
# 8. Persist results
# --------------------------------------------------------------------------
out = {
    "missingness_threshold": KEEP_MISS_THRESH,
    "distance_threshold": THRESH,
    "n_features_total": len(predictors_all),
    "n_features_after_missing": len(kept),
    "n_clusters": int(n_clusters),
    "max_abs_r_in_selected": max_r_selected,
    "dropped_for_missingness": dropped_miss,
    "representatives": representatives,
    "cluster_summary": cluster_summary,
    "manual_selection": manual_18,
    "in_both": in_both,
    "only_manual": only_manual,
    "only_dendro": only_dendro,
    "manual_to_cluster": manual_to_cluster,
}
with open(os.path.join(ROOT, "results_feature_selection.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(f"Results JSON: {os.path.join(ROOT, 'results_feature_selection.json')}")

# --------------------------------------------------------------------------
# 9. Pretty-print cluster table
# --------------------------------------------------------------------------
print("\n=== Cluster membership ===")
for c in cluster_summary:
    star = " <-- rep"
    members_str = ", ".join(
        f"{m}{star if m == c['representative'] else ''}" for m in c["members"]
    )
    print(f"[Cluster {c['cluster']:2d}, size={c['size']:2d}] {members_str}")
