# Deteksi Gambar AI-Generated vs Real Images
### Klasifikasi Citra Berbasis Deep Learning menggunakan Convolutional Neural Network (CNN)

---

## Deskripsi Proyek

Proyek ini membangun sistem klasifikasi otomatis untuk membedakan antara **gambar asli (Real Images)** dan **gambar hasil generasi AI (AI-Generated Images)** menggunakan pendekatan Deep Learning berbasis Convolutional Neural Network (CNN).

Perkembangan model generatif berbasis AI telah memungkinkan pembuatan gambar sintetis yang sangat realistis, sehingga diperlukan sistem deteksi yang andal dan akurat.

---

## Tujuan

- Membangun sistem klasifikasi citra berbasis Deep Learning
- Menggunakan CNN untuk mendeteksi gambar AI-Generated vs Real
- Menganalisis performa model melalui evaluasi kuantitatif dan visual
- Mengembangkan pipeline klasifikasi citra yang efisien dan dapat direproduksi

---

## Struktur Notebook

```
├── 1. Import Library
├── 2. Sumber / Akuisisi Data         ← HuggingFace Datasets
├── 3. Exploratory Data Analysis (EDA)
│   ├── Struktur Data
│   ├── Distribusi Kelas
   └── Visualisasi Sampel
├── 4. Data Preparation
│   ├── Split Train / Val / Test
│   ├── Preprocessing (Grayscale, Resize, Normalisasi)
│   └── Pipeline tf.data (Batching & Prefetch)
├── 5. Membangun Model CNN
├── 6. Training Model
├── 7. Evaluasi Model
│   ├── Test Accuracy & Loss
│   ├── Confusion Matrix
│   └── Classification Report
├── 8. Visualisasi Hasil Prediksi
├── 9. Inferensi Gambar Baru (via path file)
├── 10. Uji Coba Gambar Kamera Pribadi vs AI
└── 11. Kesimpulan & Saran
```

---

##  Dataset

Dataset berasal dari **HuggingFace Datasets**:

```
Hemg/AI-Generated-vs-Real-Images-Datasets
```

Berisi dua kelas:
| Label | Keterangan |
|-------|------------|
| `0`   | Real Images (gambar asli) |
| `1`   | AI-Generated Images (gambar buatan AI) |

---

##  Tech Stack & Library

| Library | Kegunaan |
|---------|----------|
| `TensorFlow / Keras` | Membangun dan melatih model CNN |
| `NumPy` | Manipulasi array dan preprocessing |
| `Matplotlib / Seaborn` | Visualisasi data dan hasil evaluasi |
| `scikit-learn` | Confusion matrix dan classification report |
| `Pillow (PIL)` | Preprocessing gambar baru |
| `HuggingFace Datasets` | Akuisisi dataset |
| `pandas` | Tabulasi hasil inferensi batch |

---

##  Preprocessing

Setiap gambar diproses melalui pipeline berikut:

1. Konversi ke mode RGB
2. Konversi ke **Grayscale**
3. Resize ke ukuran **64 × 64 piksel**
4. Normalisasi nilai piksel ke rentang **[0, 1]**
5. Penambahan dimensi channel → shape akhir: `(64, 64, 1)`

---

## Arsitektur Model CNN

```
Input (64, 64, 1)
     ↓
Conv2D(32, 3×3, ReLU) → MaxPooling2D
     ↓
Conv2D(64, 3×3, ReLU) → MaxPooling2D
     ↓
Flatten
     ↓
Dense(128, ReLU)
     ↓
Dropout(0.5)
     ↓
Dense(1, Sigmoid)  ← Output probabilitas biner
```

**Konfigurasi Training:**
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Metric: `Accuracy`
- Max Epochs: `20`
- Early Stopping: `patience=5`, monitor `val_loss`
- Batch Size: `32`

---

##  Evaluasi

Model dievaluasi menggunakan:
- **Test Accuracy & Loss**
- **Confusion Matrix** (heatmap)
- **Classification Report** (Precision, Recall, F1-Score)

---

## Cara Menjalankan

> Direkomendasikan menggunakan **Google Colab** (sudah terkonfigurasi untuk HuggingFace login dan Google Drive mount).

### 1. Clone / Buka Notebook

Buka file `.ipynb` di Google Colab atau Jupyter Notebook.

### 2. Install Dependencies

```bash
pip install datasets huggingface_hub tensorflow scikit-learn matplotlib seaborn pillow pandas
```

### 3. Login HuggingFace

```python
from huggingface_hub import login
login('YOUR_HF_TOKEN')  # Ganti dengan token HuggingFace Anda
```

### 4. Jalankan Semua Cell Secara Berurutan

Ikuti urutan cell dari atas ke bawah:
- Akuisisi data → EDA → Preprocessing → Training → Evaluasi → Inferensi

### 5. Inferensi Gambar Baru

Untuk menguji gambar Anda sendiri:
```python
image_path = "/content/nama_gambar.jpg"  # Sesuaikan path
```
Upload gambar ke Colab, lalu jalankan cell pada bagian **"Input dan Analisis Gambar Baru"**.

---

## Struktur Folder (untuk Uji Coba Batch)

Untuk menjalankan pengujian batch 10 gambar:

```
Google Drive/
└── tubesML/
    └── gambar/
        ├── gambar_1.jpg
        ├── gambar_2.jpg
        └── ...
```

Mount Google Drive dan pastikan path sesuai:
```python
folder_path = "/content/drive/MyDrive/tubesML/gambar"
```

---

## Kesimpulan

Model CNN berhasil dilatih dan diimplementasikan untuk mengklasifikasikan gambar sebagai **AI-Generated** atau **Real**, lengkap dengan **confidence score** sebagai tingkat keyakinan prediksi.

---

## Saran Pengembangan

- Menambah jumlah dataset agar model lebih generalisasi
- Menyeimbangkan distribusi kelas untuk mengurangi bias
- Menggunakan arsitektur lebih kompleks: **VGG16**, **ResNet**, atau **EfficientNet** (Transfer Learning)
- Pengujian pada lebih banyak gambar eksternal untuk evaluasi yang lebih komprehensif

---

##  Tim

> Proyek ini merupakan Tugas Besar (Tubes) mata kuliah Machine Learning 
- Muhammad Eka Dimas Saputra
- Ni Luh Made Sri Utami Pradnyandari
- Safa Ayu Artanti
- Mokhammad Bahauddin
- Tsaqillah ali
- Firanitsy Shania Karwur
---


Link Google Colab: https://colab.research.google.com/drive/1lCGa4wpiRoYuXwkjU61mfU3z4SDaydUq?usp=sharing 
