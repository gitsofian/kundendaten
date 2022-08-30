import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Lade Datensatz
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))
df = pd.read_csv(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'Mall_Customers.csv'))

# Zeige die ersten 5 Zeilen in der Konsole
# print(df.head())

# Wähle Features
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features_real = df[col_names].values


# Aufgabe a

# Annual Income
print("\nAnnual Income:")
print(f"Mittelwert: {np.mean(features_real[:, 0]):.2f} (k$)")
print(
    f"Standardabweichung: {np.std(features_real[:, 0]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 1])} (k$)")
print(f"Größten Wert: {np.max(features_real[:, 1])} (k$)")

# Age
print("\nAge:")
print(f"Mittelwert:{np.mean(features_real[:, 1]):.2f} Jahre")
print(f"Standardabweichung:{np.std(features_real[:, 1]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 1])} Jahre")
print(f"Größten Wert: {np.max(features_real[:, 1])} Jahre")

# Spending Score
print("\nSpending Score (1-100):")
print(f"Mittelwert:{np.mean(features_real[:, 2]):.2f}")
print(f"Standardabweichung:{np.std(features_real[:, 2]):.2f}")
print(f"Kleinesten Wert: {np.min(features_real[:, 2])}")
print(f"Größten Wert: {np.max(features_real[:, 2])}")

# Aufgabe b
# Kovarianzmatrix
print("\n Kovarianzmatrix Annual Incode & Age")
features_real_transposed = np.transpose(features_real)
kovarianzmat = np.cov(features_real_transposed)

print(kovarianzmat)

# Aufgabe c
# Standardskalierung

scaler = StandardScaler()
print(scaler.fit(features_real))
print(f"Mittelwert:\t {scaler.mean_}")
feature_real_scaled = scaler.transform(features_real)
print(f"Standardskalierung: {feature_real_scaled}")
print("\n")


# Aufgabe d
# Kovarianzmatrix

print("Kovarianzmatrix vor der Standardskalierung gerechnet:")
print(kovarianzmat)
print("\nKovarianzmatrix nach der Standardskalierung gerechnet:")
print(np.cov(np.transpose(feature_real_scaled)))

# Aufgabe E, F

# Set up a figure twice as tall as it is wide
# print(plt.figaspect(48/9))
fig0 = plt.figure(figsize=(20, 10))
fig0.suptitle("K-Means Clustering")

# fig.set_size_inches(20, 10)

# K-Means Object
# Daten zur Auswahl entwieder unbearbeitet oder skaliert Daten
# X = features_real           # real data
X = feature_real_scaled     # Skalierungdaten

print(f"K-Means Cluster's Center:")

inertias = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]

for n_clusters in range_n_clusters:

    fig1 = plt.figure(figsize=(20, 10))
    ax11 = fig1.add_subplot(1, 2, 1)
    # fig1, (ax11, ax12) = plt.subplots(1, 2)
    # fig1.set_size_inches(20, 10)

    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    inertias.append(kmeans.inertia_)
    cluster_labels = kmeans.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        f"For n_clusters = {n_clusters}, The average silhouette_score is: {silhouette_avg:.4f}")

    y_lower = 10

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        # sorting
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax11.fill_betweenx(np.arange(
            y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax11.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax11.set_title("The silhouette plot for the various clusters.")
    ax11.set_xlabel("The silhouette coefficient values")
    ax11.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax11.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax11.set_yticks([])  # Clear the yaxis labels / ticks
    ax11.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    ax12 = fig1.add_subplot(1, 2, 2, projection='3d')
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax12.scatter(
        X[:, 0], X[:, 1], X[:, 2], marker="s", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax12.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax12.scatter(c[0], c[1], c[2], marker="$%d$" %
                     i, alpha=1, s=50, edgecolor="k")

    ax12.set_title("The visualization of the clustered data.")
    ax12.set_xlabel("Feature space for the 1st feature")
    ax12.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


# 3D-Plot
ax3d = fig0.add_subplot(1, 2, 1, projection='3d')
ax3d.scatter(X[:, 0], X[:, 1],
             X[:, 2], c=cluster_labels, marker='s')

ax3d.set_title("Scatting Feature Data ", c="C0")
ax3d.set_xlabel("Annual Income in (k$)")
ax3d.set_ylabel("Age in Year")

# Add Elbow Plot
ax2d = fig0.add_subplot(1, 2, 2)
ax2d.plot(range_n_clusters, inertias)
ax2d.set_title("Elbow-Plot")
ax2d.set_ylabel("Inertia")
ax2d.set_xlabel("number of cluster")
ax2d.grid(True)


plt.show()
