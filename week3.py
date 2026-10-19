import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from fcmeans import FCM

# ==========================
# LOAD DATA
# ==========================
import os
df = pd.read_csv(r"C:\Users\c3698852\Desktop\Mall_Customers.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# ==========================
# FEATURE SELECTION
# ==========================

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

# ==========================
# SCALING
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)

# ==========================
# ELBOW METHOD
# ==========================

ks = range(2, 11)

inertias = []
silhouettes = []

for k in ks:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouettes.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )

# Plot Elbow + Silhouette

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(ks, inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

axes[1].plot(ks, silhouettes, marker="o")
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()

results = pd.DataFrame({
    "K": list(ks),
    "Inertia": inertias,
    "Silhouette": silhouettes
})

print("\nK Analysis")
print(results)

# ==========================
# K-MEANS
# ==========================

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Profiles:")
print(
    df.groupby("KMeans_Cluster")[features]
    .mean()
    .round(2)
)

print("\nCluster Counts:")
print(
    df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
)

# Scatter Plot

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="KMeans_Cluster",
    palette="tab10"
)

plt.title("K-Means Customer Segments")
plt.show()

# ==========================
# PCA VISUALISATION
# ==========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["KMeans_Cluster"]
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10"
)

plt.title("K-Means PCA View")
plt.show()

# ==========================
# FUZZY C-MEANS
# ==========================

fcm = FCM(
    n_clusters=k,
    random_state=42
)

fcm.fit(X_scaled)

U = fcm.u

print("\nMembership Matrix Shape:")
print(U.shape)

print("\nFirst Five Membership Scores:")
print(np.round(U[:5], 3))

df["FCM_Cluster"] = np.argmax(U, axis=1)

df["FCM_Max_Membership"] = np.max(U, axis=1)

sorted_u = np.sort(U, axis=1)

df["FCM_Membership_Gap"] = (
    sorted_u[:, -1] - sorted_u[:, -2]
)

print("\nFCM Profiles:")
print(
    df.groupby("FCM_Cluster")[features]
    .mean()
    .round(2)
)

# ==========================
# FCM VISUALISATION
# ==========================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="FCM_Cluster",
    palette="tab10"
)

plt.title("Fuzzy C-Means Segments")
plt.show()

# ==========================
# MEMBERSHIP CONFIDENCE
# ==========================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["FCM_Max_Membership"],
    bins=20,
    kde=True
)

plt.title("Maximum Membership Distribution")
plt.xlabel("Maximum Membership")

plt.show()

# ==========================
# AMBIGUOUS CUSTOMERS
# ==========================

print("\nMost Ambiguous Customers")

ambiguous = df.nsmallest(
    10,
    "FCM_Max_Membership"
)

print(
    ambiguous[
        features +
        [
            "FCM_Cluster",
            "FCM_Max_Membership"
        ]
    ]
)

# ==========================
# K COMPARISON
# ==========================

comparison = []

for candidate_k in [3, 4, 5, 6]:

    model = FCM(
        n_clusters=candidate_k,
        random_state=42
    )

    model.fit(X_scaled)

    U_k = model.u

    memberships = np.max(U_k, axis=1)

    sorted_memberships = np.sort(
        U_k,
        axis=1
    )

    gap = (
        sorted_memberships[:, -1]
        - sorted_memberships[:, -2]
    )

    comparison.append({
        "K": candidate_k,
        "Mean Max Membership":
            memberships.mean(),
        "Median Max Membership":
            np.median(memberships),
        "Mean Gap":
            gap.mean()
    })

comparison_df = pd.DataFrame(
    comparison
).round(3)

print("\nFCM Comparison")
print(comparison_df)

# ==========================
# DBSCAN
# ==========================

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.7,
    min_samples=5
)

db_labels = dbscan.fit_predict(X_scaled)

print("\nDBSCAN Results:")
print(
    pd.Series(db_labels)
    .value_counts()
    .sort_index()
)

# ==========================
# HIERARCHICAL CLUSTERING
# ==========================

from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

hier_labels = hierarchical.fit_predict(
    X_scaled
)

print("\nHierarchical Clustering Results:")
print(
    pd.Series(hier_labels)
    .value_counts()
    .sort_index()
)

# ==========================
# SAVE RESULTS
# ==========================

df.to_csv(
    "Customer_Segmentation_Results.csv",
    index=False
)

print(
    "\nAnalysis Complete!"
)
print(
    "Results saved to: Customer_Segmentation_Results.csv"
)