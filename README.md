# 📊 AI & Data Science Analytics Hub

Aplikasi web interaktif berbasis **Python & Streamlit** untuk analisis eksplorasi data (EDA), pelatihan model *Machine Learning* otomatis (Klasifikasi & Regresi), pencatatan eksperimen ke database **SQLite**, serta impor/ekspor data **CSV & Excel (.xlsx)**.

## ✨ Fitur Utama
- **Multi-Format Input & Output**: Mendukung unggah dan ekspor data ke format `.csv` maupun `.xlsx` (Excel).
- **Automated Machine Learning**: Pelatihan model *Random Forest* (Klasifikasi & Regresi) dengan kustomisasi *hyperparameter* langsung dari antarmuka.
- **SQLite Experiment Tracking**: Setiap pelatihan model secara otomatis menyimpan *timestamp*, parameter, dan skor akurasi/R2 ke dalam database lokal `ml_experiments.db`.
- **CI/CD GitHub Actions**: Terintegrasi dengan pengecekan kualitas kode otomatis (`flake8` & *smoke test*) setiap kali ada perubahan kode.

## 🚀 Cara Menjalankan di Lokal
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/username-anda/ml-analytics-hub.git](https://github.com/username-anda/ml-analytics-hub.git)
   cd ml-analytics-hub
