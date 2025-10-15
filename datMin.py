import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Kluster ChatBot AI",
    page_icon="📊",
    layout="wide"
)

# --- Bagian 1: Fungsi-fungsi Manual (Scaler, PCA, K-Means) ---

def euclidean_distance(point1, point2):
    """Menghitung jarak Euclidean."""
    return np.sqrt(np.sum((point1 - point2)**2))

def manual_scaler(data):
    """Melakukan scaling data (Standard Scaler) secara manual."""
    data_np = data.to_numpy()
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0)
    std[std == 0] = 1
    return (data_np - mean) / std

def manual_pca(X, n_components):
    """Melakukan Principal Component Analysis (PCA) secara manual."""
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Konversi ke real values untuk menghindari complex warning
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    projection_matrix = sorted_eigenvectors[:, :n_components]
    return X.dot(projection_matrix)

class KMeansManual:
    """Implementasi K-Means Manual yang juga menghitung Inertia."""
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = []
        self.inertia_ = 0

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            clusters = self._assign_clusters(X)
            self._update_centroids(X, clusters)
            if np.all(old_centroids == self.centroids):
                break
        
        self._calculate_inertia(X, self._get_cluster_labels(X))
        return self._get_cluster_labels(X)

    def _assign_clusters(self, X):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(X):
            distances = [euclidean_distance(sample, point) for point in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(idx)
        return clusters

    def _update_centroids(self, X, clusters):
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:
                self.centroids[cluster_idx] = np.mean(X[cluster], axis=0)
    
    def _get_cluster_labels(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        for idx, sample in enumerate(X):
            distances = [euclidean_distance(sample, point) for point in self.centroids]
            labels[idx] = np.argmin(distances)
        return labels

    def _calculate_inertia(self, X, labels):
        self.inertia_ = 0
        for i, sample in enumerate(X):
            centroid_idx = labels[i]
            self.inertia_ += euclidean_distance(sample, self.centroids[centroid_idx])**2

# --- Bagian 2: Proses Utama ---

# Judul Aplikasi
st.title("Analisis Kluster Pengguna ChatBot AI")
st.markdown("---")

# Sidebar untuk kontrol
st.sidebar.header("Pengaturan Analisis")
st.sidebar.markdown("Aplikasi ini menggunakan implementasi manual untuk K-Means, PCA, dan Standard Scaler")

# Upload atau gunakan data default
data_source = st.sidebar.radio("Sumber Data:", ["Gunakan Data Default", "Upload Data"])

if data_source == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Silakan upload file CSV atau gunakan data default")
        st.stop()
else:
    df = pd.read_csv("Data.csv")

# Tampilkan data
with st.expander("Lihat Data"):
    st.dataframe(df.head(10))
    st.write(f"Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")

# Fitur yang digunakan - dibagi menjadi 2 kelompok
use_features = [
    'freq_overall', 'use_brainstorming', 'use_outlining', 'use_drafting',
    'use_summarizing', 'use_grammar', 'use_coding', 'use_explaining',
    'use_referencing'
]

perception_features = [
    'perception_improves_learning', 'perception_hinders_crit_think', 
    'perception_plagiarism_concern', 'perception_accuracy', 
    'perception_essential_tool'
]

features = use_features + perception_features



# Parameter untuk metode elbow
st.sidebar.markdown("---")
max_k = st.sidebar.slider("Maksimal K untuk Metode Elbow", 5, 15, 10)


# Analisis terpisah untuk Use dan Perception
st.header("ANALISIS KLUSTER TERPISAH")
st.markdown("Dua analisis kluster independen: satu untuk Use Features, satu untuk Perception Features")

# ========== ANALISIS USE FEATURES ==========
st.markdown("---")
st.subheader("A. KLUSTER USE FEATURES (Pola Penggunaan)")

data_use = df[use_features]
scaled_use = manual_scaler(data_use)

# Elbow Method untuk Use
st.markdown("**1. Metode Elbow - Use Features**")
with st.spinner("Menghitung WCSS untuk Use Features..."):
    wcss_use = []
    k_range = range(1, max_k + 1)
    progress_bar = st.progress(0)
    for idx, k in enumerate(k_range):
        kmeans = KMeansManual(n_clusters=k, random_state=42)
        kmeans.fit(scaled_use)
        wcss_use.append(kmeans.inertia_)
        progress_bar.progress((idx + 1) / len(k_range) / 2)
    progress_bar.empty()

fig_use, ax_use = plt.subplots(figsize=(10, 5))
ax_use.plot(k_range, wcss_use, 'bo-')
ax_use.set_xlabel('Jumlah Kluster (k)', fontsize=12)
ax_use.set_ylabel('WCSS (Inertia)', fontsize=12)
ax_use.set_title('Metode Elbow - Use Features', fontsize=14)
ax_use.grid(True, alpha=0.3)
ax_use.set_xticks(k_range)
ax_use.axvline(x=3, color='r', linestyle='--', label='Elbow Point (k=3)', linewidth=2)
ax_use.legend()
st.pyplot(fig_use)
plt.close()

optimal_k_use = st.sidebar.selectbox("Jumlah Kluster Use Features", list(range(2, max_k + 1)), index=1, key="k_use")

# Klustering Use
st.markdown(f"**2. Hasil Klustering Use Features (k={optimal_k_use})**")
with st.spinner(f"Menjalankan K-Means untuk Use Features..."):
    kmeans_use = KMeansManual(n_clusters=optimal_k_use, max_iter=300, random_state=42)
    cluster_labels_use = kmeans_use.fit(scaled_use)
    df['cluster_use'] = cluster_labels_use

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Data Points", len(df))
with col2:
    st.metric("Inertia (WCSS)", f"{kmeans_use.inertia_:.2f}")

# Distribusi kluster Use
cluster_counts_use = df['cluster_use'].value_counts().sort_index()
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts_use.plot(kind='bar', ax=ax, color=sns.color_palette("Blues", n_colors=optimal_k_use))
    ax.set_xlabel('Kluster Use', fontsize=12)
    ax.set_ylabel('Jumlah Data', fontsize=12)
    ax.set_title('Distribusi Kluster Use Features', fontsize=14)
    ax.set_xticklabels([f'Use-{i}' for i in cluster_counts_use.index], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("**Distribusi:**")
    for idx in cluster_counts_use.index:
        percentage = (cluster_counts_use[idx] / len(df)) * 100
        st.write(f"Use-{idx}: {cluster_counts_use[idx]} ({percentage:.1f}%)")

# Profil kluster Use
cluster_profile_use = df.groupby('cluster_use')[use_features].mean()
st.write("**Profil Rata-rata Kluster Use:**")
st.dataframe(cluster_profile_use.round(2).style.background_gradient(cmap='Blues', axis=1))

# ========== ANALISIS PERCEPTION FEATURES ==========
st.markdown("---")
st.subheader("B. KLUSTER PERCEPTION FEATURES (Persepsi)")

data_perception = df[perception_features]
scaled_perception = manual_scaler(data_perception)

# Elbow Method untuk Perception
st.markdown("**1. Metode Elbow - Perception Features**")
with st.spinner("Menghitung WCSS untuk Perception Features..."):
    wcss_perception = []
    progress_bar = st.progress(0.5)
    for idx, k in enumerate(k_range):
        kmeans = KMeansManual(n_clusters=k, random_state=42)
        kmeans.fit(scaled_perception)
        wcss_perception.append(kmeans.inertia_)
        progress_bar.progress(0.5 + (idx + 1) / len(k_range) / 2)
    progress_bar.empty()

fig_perc, ax_perc = plt.subplots(figsize=(10, 5))
ax_perc.plot(k_range, wcss_perception, 'ro-')
ax_perc.set_xlabel('Jumlah Kluster (k)', fontsize=12)
ax_perc.set_ylabel('WCSS (Inertia)', fontsize=12)
ax_perc.set_title('Metode Elbow - Perception Features', fontsize=14)
ax_perc.grid(True, alpha=0.3)
ax_perc.set_xticks(k_range)
ax_perc.axvline(x=3, color='darkred', linestyle='--', label='Elbow Point (k=3)', linewidth=2)
ax_perc.legend()
st.pyplot(fig_perc)
plt.close()

optimal_k_perception = st.sidebar.selectbox("Jumlah Kluster Perception Features", list(range(2, max_k + 1)), index=1, key="k_perception")

# Klustering Perception
st.markdown(f"**2. Hasil Klustering Perception Features (k={optimal_k_perception})**")
with st.spinner(f"Menjalankan K-Means untuk Perception Features..."):
    kmeans_perception = KMeansManual(n_clusters=optimal_k_perception, max_iter=300, random_state=42)
    cluster_labels_perception = kmeans_perception.fit(scaled_perception)
    df['cluster_perception'] = cluster_labels_perception

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Data Points", len(df))
with col2:
    st.metric("Inertia (WCSS)", f"{kmeans_perception.inertia_:.2f}")

# Distribusi kluster Perception
cluster_counts_perception = df['cluster_perception'].value_counts().sort_index()
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts_perception.plot(kind='bar', ax=ax, color=sns.color_palette("Reds", n_colors=optimal_k_perception))
    ax.set_xlabel('Kluster Perception', fontsize=12)
    ax.set_ylabel('Jumlah Data', fontsize=12)
    ax.set_title('Distribusi Kluster Perception Features', fontsize=14)
    ax.set_xticklabels([f'Perc-{i}' for i in cluster_counts_perception.index], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("**Distribusi:**")
    for idx in cluster_counts_perception.index:
        percentage = (cluster_counts_perception[idx] / len(df)) * 100
        st.write(f"Perc-{idx}: {cluster_counts_perception[idx]} ({percentage:.1f}%)")

# Profil kluster Perception
cluster_profile_perception = df.groupby('cluster_perception')[perception_features].mean()
st.write("**Profil Rata-rata Kluster Perception:**")
st.dataframe(cluster_profile_perception.round(2).style.background_gradient(cmap='Reds', axis=1))


# --- Bagian 5: Visualisasi ---
st.markdown("---")
st.header("3. Visualisasi")


# Radar Chart Use
st.subheader("A. Radar Chart - Use Features")
fig_radar_use = go.Figure()
colors_blue = ['rgb(46, 134, 193)', 'rgb(52, 152, 219)', 'rgb(93, 173, 226)', 
               'rgb(133, 193, 233)', 'rgb(174, 214, 241)']

for i, row in cluster_profile_use.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    fig_radar_use.add_trace(go.Scatterpolar(
        r=values,
        theta=use_features + [use_features[0]],
        fill='toself',
        name=f'Use-{i}',
        line_color=colors_blue[i % len(colors_blue)]
    ))

fig_radar_use.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max(cluster_profile_use.max())])),
    showlegend=True,
    title="Profil Kluster Use Features",
    height=600
)
st.plotly_chart(fig_radar_use, use_container_width=True)

# Radar Chart Perception
st.subheader("B. Radar Chart - Perception Features")
fig_radar_perc = go.Figure()
colors_red = ['rgb(231, 76, 60)', 'rgb(241, 148, 138)', 'rgb(245, 183, 177)', 
              'rgb(250, 219, 216)', 'rgb(192, 57, 43)']

for i, row in cluster_profile_perception.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    fig_radar_perc.add_trace(go.Scatterpolar(
        r=values,
        theta=perception_features + [perception_features[0]],
        fill='toself',
        name=f'Perc-{i}',
        line_color=colors_red[i % len(colors_red)]
    ))

fig_radar_perc.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max(cluster_profile_perception.max())])),
    showlegend=True,
    title="Profil Kluster Perception Features",
    height=600
)
st.plotly_chart(fig_radar_perc, use_container_width=True)

# PCA Scatter Plot
st.subheader("C. Scatter Plot PCA")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**PCA - Use Features**")
    pca_use = manual_pca(scaled_use, n_components=2)
    pca_df_use = pd.DataFrame(data=pca_use, columns=['PC1', 'PC2'])
    pca_df_use['cluster'] = cluster_labels_use
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='deep',
                   data=pca_df_use, legend="full", s=80, alpha=0.7, ax=ax)
    
    centroids_use_pca = manual_pca(kmeans_use.centroids, n_components=2)
    ax.scatter(centroids_use_pca[:, 0], centroids_use_pca[:, 1],
              s=250, c='darkblue', marker='X', label='Centroids', 
              edgecolors='black', linewidths=2)
    
    ax.set_title('Kluster Use Features (PCA)', fontsize=14)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Cluster')
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("**PCA - Perception Features**")
    pca_perception = manual_pca(scaled_perception, n_components=2)
    pca_df_perception = pd.DataFrame(data=pca_perception, columns=['PC1', 'PC2'])
    pca_df_perception['cluster'] = cluster_labels_perception
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='deep',
                   data=pca_df_perception, legend="full", s=80, alpha=0.7, ax=ax)
    
    centroids_perc_pca = manual_pca(kmeans_perception.centroids, n_components=2)
    ax.scatter(centroids_perc_pca[:, 0], centroids_perc_pca[:, 1],
              s=250, c='darkred', marker='X', label='Centroids', 
              edgecolors='black', linewidths=2)
    
    ax.set_title('Kluster Perception Features (PCA)', fontsize=14)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Cluster')
    st.pyplot(fig)
    plt.close()



# Insight dan Interpretasi
st.markdown("---")
st.header("4. Interpretasi dan Insight")

st.markdown("### Karakteristik Kluster Use Features:")
for cluster_idx in range(optimal_k_use):
    cluster_data = cluster_profile_use.loc[cluster_idx]
    top_features = cluster_data.nlargest(3)
    
    st.markdown(f"**Use-{cluster_idx}** ({cluster_counts_use[cluster_idx]} pengguna - {(cluster_counts_use[cluster_idx]/len(df)*100):.1f}%):")
    st.write("Ciri khas (Top 3):")
    for feat, val in top_features.items():
        st.write(f"- {feat}: **{val:.2f}**")
    st.write("")

st.markdown("### Karakteristik Kluster Perception Features:")
for cluster_idx in range(optimal_k_perception):
    cluster_data = cluster_profile_perception.loc[cluster_idx]
    top_features = cluster_data.nlargest(3)
    
    st.markdown(f"**Perc-{cluster_idx}** ({cluster_counts_perception[cluster_idx]} pengguna - {(cluster_counts_perception[cluster_idx]/len(df)*100):.1f}%):")
    st.write("Ciri khas (Top 3):")
    for feat, val in top_features.items():
        st.write(f"- {feat}: **{val:.2f}**")
    st.write("")

