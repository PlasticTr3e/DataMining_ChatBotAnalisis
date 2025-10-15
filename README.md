# Analisis Kluster Pengguna ChatBot AI

Aplikasi analisis kluster untuk menganalisis pola penggunaan dan persepsi pengguna terhadap ChatBot AI menggunakan implementasi manual K-Means, PCA, dan Standard Scaler.

## Fitur Utama

- **Analisis Kluster Terpisah**: Analisis independen untuk Use Features dan Perception Features
- **Implementasi Manual**: Semua algoritma (K-Means, PCA, Standard Scaler) diimplementasikan tanpa scikit-learn
- **Visualisasi Interaktif**: Radar chart, scatter plot PCA, dan heatmap
- **Metode Elbow**: Untuk menentukan jumlah kluster optimal
- **Download Hasil**: Export data dan profil kluster dalam format CSV

## Persyaratan Sistem

- Python 3.7 atau lebih baru
- Library yang diperlukan (lihat `requirements.txt`)

## Instalasi

1. Clone atau download repository ini
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Cara Penggunaan

### 1. Menjalankan Aplikasi

```bash
streamlit run datMin.py
```

Aplikasi akan terbuka di browser pada `localhost`

### 2. Upload Data

Di sidebar, pilih:
- **Gunakan Data Default**: Menggunakan file `Data.csv` yang sudah ada
- **Upload Data**: Upload file CSV Anda sendiri

Format data harus memiliki kolom:
- **Use Features**: `freq_overall`, `use_brainstorming`, `use_outlining`, `use_drafting`, `use_summarizing`, `use_grammar`, `use_coding`, `use_explaining`, `use_referencing`
- **Perception Features**: `perception_improves_learning`, `perception_hinders_crit_think`, `perception_plagiarism_concern`, `perception_accuracy`, `perception_essential_tool`

### 3. Konfigurasi Analisis

**Di Sidebar:**

1. **Maksimal K untuk Metode Elbow** (5-15): 
   - Tentukan rentang jumlah kluster yang akan diuji
   - Rekomendasi: 10 untuk dataset sedang

2. **Jumlah Kluster Use Features** (2-max_k):
   - Pilih jumlah kluster untuk analisis Use Features
   - Lihat grafik Elbow untuk menentukan nilai optimal

3. **Jumlah Kluster Perception Features** (2-max_k):
   - Pilih jumlah kluster untuk analisis Perception Features
   - Lihat grafik Elbow untuk menentukan nilai optimal

### 4. Membaca Hasil Analisis

#### A. KLUSTER USE FEATURES (Pola Penggunaan)

**1. Metode Elbow**
- Grafik menunjukkan WCSS (Within-Cluster Sum of Squares) untuk setiap nilai k
- Pilih nilai k di titik elbow point dimana penurunan WCSS mulai melambat

**2. Hasil Klustering**
- **Distribusi Kluster**: Bar chart menunjukkan jumlah data di setiap kluster
- **Profil Kluster**: Tabel rata-rata nilai setiap fitur per kluster
  - Nilai lebih tinggi = penggunaan lebih intensif
  - Warna lebih gelap = nilai lebih tinggi

#### B. KLUSTER PERCEPTION FEATURES (Persepsi)

**1. Metode Elbow**
- Similar dengan Use Features, tapi untuk fitur persepsi

**2. Hasil Klustering**
- **Distribusi Kluster**: Jumlah pengguna di setiap kelompok persepsi
- **Profil Kluster**: Rata-rata nilai persepsi per kluster
  - Nilai lebih tinggi = persepsi lebih positif/kuat

### 5. Visualisasi

#### Radar Chart
- **Use Features**: Menampilkan profil penggunaan setiap kluster
- **Perception Features**: Menampilkan profil persepsi setiap kluster
- Semakin luas area = nilai rata-rata lebih tinggi

#### Scatter Plot PCA
- **2D visualization** dari data multidimensional
- Setiap titik = satu responden
- Warna berbeda = kluster berbeda
- Simbol X merah = centroid (pusat kluster)

#### Heatmap Hubungan
- Menampilkan **crosstab** antara kluster Use dan Perception
- Angka menunjukkan jumlah responden di kombinasi kluster tertentu
- Warna lebih gelap = jumlah lebih banyak

### 6. Interpretasi dan Insight

Bagian ini menampilkan:
- **Top 3 karakteristik** setiap kluster Use
- **Top 3 karakteristik** setiap kluster Perception
- **Hubungan** antara kluster Use dan Perception

**Contoh Interpretasi:**
- **Use-0** dengan `use_coding: 4.5` = Kelompok pengguna yang aktif menggunakan ChatBot untuk coding
- **Perc-1** dengan `perception_essential_tool: 4.8` = Kelompok yang sangat percaya ChatBot sebagai tool esensial

### 7. Download Hasil

Aplikasi menyediakan 3 file download:

1. **Data dengan Label Kluster** (`hasil_clustering_terpisah.csv`)
   - Dataset lengkap dengan kolom `cluster_use` dan `cluster_perception`
   - Gunakan untuk analisis lanjutan

2. **Profil Use** (`profil_cluster_use.csv`)
   - Rata-rata nilai Use Features per kluster
   - Gunakan untuk presentasi atau laporan

3. **Profil Perception** (`profil_cluster_perception.csv`)
   - Rata-rata nilai Perception Features per kluster
   - Gunakan untuk presentasi atau laporan

## Struktur File

```
DataMining_ChatBotAnalisis/
│
├── datMin.py              # Aplikasi Streamlit utama
├── Data.csv               # Dataset default
├── requirements.txt       # Dependencies Python
└── README.md             # Dokumentasi ini
```

## Algoritma yang Diimplementasikan

### 1. Standard Scaler (Manual)
```python
scaled_data = (data - mean) / std
```
Menormalisasi data agar setiap fitur memiliki skala yang sama.

### 2. PCA (Manual)
- Menghitung covariance matrix
- Eigen decomposition
- Proyeksi ke 2 komponen utama untuk visualisasi

### 3. K-Means (Manual)
- Inisialisasi centroid secara random
- Assignment: assign setiap data ke centroid terdekat
- Update: hitung ulang centroid
- Iterasi hingga konvergen atau max_iter tercapai