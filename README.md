# Analisis Sentimen dengan Support Vector Machine (SVM) menggunakan Dataset dari IndoNLU

Proyek ini melakukan analisis sentimen (klasifikasi teks) pada teks berbahasa Indonesia menggunakan algoritma **Support Vector Machine (SVM)** dengan representasi fitur **TF-IDF**. Dataset yang digunakan adalah **SmSA (Sentiment Analysis, Prosa)** dari benchmark **IndoNLU**, yang mengklasifikasikan teks ke dalam tiga label sentimen: `positive`, `negative`, dan `netral`.

Bahasa: Python (Jupyter Notebook) · Model: SVM (kernel linear) · Fitur: TF-IDF · Dataset: IndoNLU – SmSA

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Struktur Repository](#struktur-repository)
- [Dataset](#dataset)
- [Alur Kerja](#alur-kerja)
- [Instalasi dan Menjalankan](#instalasi-dan-menjalankan)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Catatan dan Potensi Pengembangan](#catatan-dan-potensi-pengembangan)
- [Referensi](#referensi)

## Tentang Proyek

Notebook `code.ipynb` berisi eksperimen end-to-end untuk menganalisis sentimen dari teks berbahasa Indonesia, mencakup:

1. Eksplorasi data (EDA) — distribusi label, panjang teks, dan visualisasi word cloud.
2. Ekstraksi fitur menggunakan `TfidfVectorizer`.
3. Pelatihan model klasifikasi menggunakan `SVM` (kernel linear) dari `scikit-learn`.
4. Evaluasi model menggunakan `classification_report`.
5. Pengujian prediksi sentimen pada kalimat baru.

## Struktur Repository

```
.
├── code.ipynb      # Notebook utama berisi seluruh eksperimen
└── indonlu/        # Referensi ke dataset IndoNLU (lihat catatan di bawah)
```

> **Catatan:** folder `indonlu` terdaftar sebagai *git submodule* yang mengarah ke repository [IndoNLP/indonlu](https://github.com/IndoNLP/indonlu), namun repo ini belum memiliki file `.gitmodules`. Akibatnya, folder tersebut akan **kosong** saat repo di-clone secara biasa. Notebook sudah menangani ini secara otomatis lewat perintah `!git clone https://github.com/indobenchmark/indonlu` pada sel kedua, sehingga dataset tetap bisa didapatkan saat notebook dijalankan dari awal (lihat bagian [Instalasi](#instalasi-dan-menjalankan)).

## Dataset

Dataset yang dipakai adalah **SmSA (Document Sentiment Prosa)**, bagian dari koleksi benchmark IndoNLU untuk NLP Bahasa Indonesia, dengan lokasi file:

```
indonlu/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv
indonlu/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv
```

Setiap baris berisi dua kolom: `Teks` (kalimat) dan `Target` (label sentimen: `positive`, `negative`, atau `netral`).

## Alur Kerja

1. **Import Library** — `pandas`, `matplotlib`, `wordcloud`, dan modul `sklearn` (`CountVectorizer`, `TfidfVectorizer`, `svm`, `classification_report`).
2. **Eksploratory Data Analysis (EDA)** — melihat jumlah data per label, distribusi panjang teks, serta word cloud untuk keseluruhan data dan khusus data dengan label negatif.
3. **Feature Engineering** — transformasi teks menjadi vektor numerik menggunakan `TfidfVectorizer` (dengan parameter `min_df=5`, `max_df=0.8`, `sublinear_tf=True`).
4. **Pemodelan** — melatih `svm.SVC(kernel='linear')` pada data latih.
5. **Evaluasi** — menampilkan `classification_report` untuk melihat precision, recall, dan f1-score per label.
6. **Uji Prediksi** — mencoba model pada kalimat baru untuk melihat prediksi sentimennya.

## Instalasi dan Menjalankan

1. Clone repository ini:
   ```bash
   git clone https://github.com/febryofibonacciamadeo/Analisis_Sentiment_dengan_Support_Vector_Machine_-SVM-_menggunakan_dataset_dari_IndoNLU.git
   cd Analisis_Sentiment_dengan_Support_Vector_Machine_-SVM-_menggunakan_dataset_dari_IndoNLU
   ```

2. Install dependency yang dibutuhkan:
   ```bash
   pip install pandas matplotlib wordcloud scikit-learn jupyter
   ```

3. Buka dan jalankan `code.ipynb` secara berurutan dari sel paling atas:
   ```bash
   jupyter notebook code.ipynb
   ```
   Sel kedua pada notebook akan menjalankan `!git clone https://github.com/indobenchmark/indonlu`, sehingga dataset IndoNLU akan otomatis terunduh ke direktori kerja saat notebook dieksekusi.

## Contoh Penggunaan

Setelah model dilatih, notebook menyediakan fungsi untuk memprediksi sentimen dari kalimat baru, misalnya:

```python
teks = 'Bahagia hatiku melihat pernikahan putri sulungku yang cantik jelita'
teks_vector = tfidfvec.transform([teks])
print(classifier_linear.predict(teks_vector))
```

## Catatan dan Potensi Pengembangan

Beberapa hal yang bisa menjadi perhatian bila proyek ini ingin dikembangkan lebih lanjut:

- **Evaluasi saat ini dilakukan pada data latih (train), bukan data validasi.** `classification_report` di notebook dihitung dari prediksi terhadap `tr_vec`/`df_tr['Target']`, sehingga metrik yang tampil mencerminkan performa pada data yang sama dengan yang dipakai untuk melatih model. Untuk mengukur kemampuan generalisasi model secara lebih representatif, sebaiknya ditambahkan evaluasi pada `te_vec` (data validasi/`df_te`).
- **Fungsi `pred_classification_svm(text)` saat ini mengacu ke variabel global `teks`, bukan parameter `text`** yang diterimanya. Akibatnya, prediksi untuk daftar kalimat baru pada sel terakhir kemungkinan tidak mencerminkan masing-masing kalimat secara individual — perlu diperbaiki agar fungsi memproses parameter `text` yang diberikan.
- **Menambahkan file `.gitmodules`** agar folder `indonlu` ter-resolve otomatis sebagai submodule saat repository di-clone, tanpa bergantung pada perintah `!git clone` di dalam notebook.
- Dapat dieksplorasi kernel SVM lain (misalnya `rbf`) atau perbandingan dengan model lain sebagai baseline tambahan.

## Referensi

- [IndoNLU Benchmark](https://github.com/IndoNLP/indonlu) — kumpulan dataset dan benchmark NLP Bahasa Indonesia.
- [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
