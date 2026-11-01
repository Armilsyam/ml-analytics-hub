import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

# ==========================================
# KONFIGURASI HALAMAN & DATABASE SQLITE
# ==========================================
st.set_page_config(
    page_title="AI & Data Science Studio",
    page_icon="📊",
    layout="wide"
)

DB_NAME = "ml_experiments.db"

def init_db():
    """Inisialisasi tabel SQLite untuk menyimpan riwayat eksperimen ML dan dataset."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            dataset_name TEXT,
            task_type TEXT,
            target_column TEXT,
            n_estimators INTEGER,
            max_depth INTEGER,
            primary_metric_name TEXT,
            primary_metric_val REAL,
            secondary_metric_name TEXT,
            secondary_metric_val REAL
        )
    """)
    conn.commit()
    conn.close()

def log_experiment(dataset_name, task_type, target_col, n_est, max_d, m1_name, m1_val, m2_name, m2_val):
    """Menyimpan hasil training model ke SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiment_logs (
            timestamp, dataset_name, task_type, target_column,
            n_estimators, max_depth, primary_metric_name, primary_metric_val,
            secondary_metric_name, secondary_metric_val
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_name, task_type, target_col, n_est, max_d,
        m1_name, round(float(m1_val), 4), m2_name, round(float(m2_val), 4)
    ))
    conn.commit()
    conn.close()

def get_experiment_logs():
    """Mengambil seluruh riwayat eksperimen dari SQLite."""
    conn = sqlite3.connect(DB_NAME)
    df_logs = pd.read_sql_query("SELECT * FROM experiment_logs ORDER BY id DESC", conn)
    conn.close()
    return df_logs

def export_to_excel(df):
    """Mengonversi DataFrame ke buffer Excel (.xlsx) untuk diunduh."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Output')
    return output.getvalue()

# Jalankan inisialisasi DB saat aplikasi mulai
init_db()

# ==========================================
# SIDEBAR: INPUT DATA (CSV / EXCEL / DEMO)
# ==========================================
st.sidebar.title("⚙️ Panel Kontrol Data")
data_source = st.sidebar.radio(
    "Pilih Sumber Data:",
    ["Upload File (CSV / Excel)", "Gunakan Dataset Sintetis (Demo)"]
)

df = None
dataset_name = "Synthetic_Demo_Data"

if data_source == "Upload File (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau Excel (.xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Berhasil memuat: {dataset_name}")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")
else:
    np.random.seed(42)
    n_samples = 300
    df = pd.DataFrame({
        "Fitur_A (Suhu)": np.random.uniform(100, 500, n_samples).round(2),
        "Fitur_B (Tekanan)": np.random.uniform(10, 100, n_samples).round(2),
        "Fitur_C (Komposisi_X)": np.random.normal(50, 15, n_samples).round(2),
        "Kategori_Material": np.random.choice(["Tipe_Alpha", "Tipe_Beta", "Tipe_Gamma"], n_samples),
    })
    # Buat target kontinu dan klasifikasi
    df["Skor_Performa"] = (
        df["Fitur_A (Suhu)"] * 0.35 +
        df["Fitur_B (Tekanan)"] * 1.2 -
        df["Fitur_C (Komposisi_X)"] * 0.5 +
        np.random.normal(0, 10, n_samples)
    ).round(2)
    df["Status_Kualitas"] = np.where(df["Skor_Performa"] > df["Skor_Performa"].median(), "Optimal", "Standar")

# ==========================================
# HALAMAN UTAMA
# ==========================================
st.title("📊 AI & Data Science Analytics Hub")
st.markdown("Platform terintegrasi untuk **Exploratory Data Analysis (EDA)**, **Pelatihan Machine Learning**, dan **Manajemen Log SQLite**.")

if df is not None:
    tab1, tab2, tab3 = st.tabs([
        "🔍 1. Eksplorasi Data & Ekspor",
        "🤖 2. Pelatihan Model ML & Prediksi",
        "🗄️ 3. Database Log Eksperimen (SQLite)"
    ])

    # ------------------------------------------
    # TAB 1: EDA & EKSPOR DATA
    # ------------------------------------------
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Baris (Sampel)", df.shape[0])
        col2.metric("Total Kolom (Fitur)", df.shape[1])
        col3.metric("Total Missing Values", int(df.isna().sum().sum()))

        st.subheader("Pratinjau Dataset")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Ringkasan Statistik")
        st.dataframe(df.describe(include='all').T, use_container_width=True)

        # Simpan ke tabel SQLite opsional & Ekspor CSV/Excel
        st.subheader("Simpan & Ekspor Dataset")
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh sebagai CSV",
                data=csv_bytes,
                file_name=f"exported_{dataset_name}.csv",
                mime="text/csv"
            )
        with c_exp2:
            excel_bytes = export_to_excel(df)
            st.download_button(
                label="⬇️ Unduh sebagai Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"exported_{dataset_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with c_exp3:
            if st.button("💾 Simpan Dataset Mentah ke Tabel SQLite"):
                conn = sqlite3.connect(DB_NAME)
                df.to_sql("saved_dataset", conn, if_exists="replace", index=False)
                conn.close()
                st.success("Dataset berhasil disimpan ke tabel `saved_dataset` di SQLite!")

    # ------------------------------------------
    # TAB 2: MACHINE LEARNING PIPELINE
    # ------------------------------------------
    with tab2:
        st.subheader("Konfigurasi Pemodelan Machine Learning")
        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            target_col = st.selectbox("Pilih Kolom Target (Y):", df.columns, index=len(df.columns)-1)
            default_features = [c for c in df.columns if c != target_col]
            feature_cols = st.multiselect("Pilih Kolom Fitur (X):", default_features, default=default_features)
            task_type = st.selectbox("Jenis Tugas ML:", ["Klasifikasi (Classification)", "Regresi (Regression)"])

        with col_cfg2:
            test_size = st.slider("Rasio Data Uji (Test Size):", 0.1, 0.5, 0.2, 0.05)
            n_estimators = st.slider("Jumlah Pohon (n_estimators):", 10, 300, 100, 10)
            max_depth = st.slider("Kedalaman Maksimal Pohon (max_depth):", 2, 20, 8, 1)

        if st.button("🚀 Latih Model & Simpan ke SQLite", type="primary"):
            if len(feature_cols) == 0:
                st.warning("Pilih minimal 1 kolom fitur (X) terlebih dahulu!")
            else:
                df_clean = df[feature_cols + [target_col]].dropna().copy()
                X = pd.get_dummies(df_clean[feature_cols], drop_first=True)
                y_raw = df_clean[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_raw, test_size=test_size, random_state=42
                )

                if task_type.startswith("Klasifikasi"):
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train.astype(str))
                    y_test_enc = le.transform(y_test.astype(str))

                    model = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                    model.fit(X_train, y_train_enc)
                    preds_enc = model.predict(X_test)
                    preds_label = le.inverse_transform(preds_enc)

                    acc = accuracy_score(y_test_enc, preds_enc)
                    f1 = f1_score(y_test_enc, preds_enc, average="weighted")

                    m1_name, m1_val = "Accuracy", acc
                    m2_name, m2_val = "F1-Score (Weighted)", f1
                    pred_output = preds_label
                else:
                    y_train_num = pd.to_numeric(y_train, errors="coerce").fillna(0)
                    y_test_num = pd.to_numeric(y_test, errors="coerce").fillna(0)

                    model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                    model.fit(X_train, y_train_num)
                    preds_num = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test_num, preds_num))
                    r2 = r2_score(y_test_num, preds_num)

                    m1_name, m1_val = "R2 Score", r2
                    m2_name, m2_val = "RMSE", rmse
                    pred_output = np.round(preds_num, 4)

                # Catat ke SQLite
                log_experiment(
                    dataset_name, task_type, target_col,
                    n_estimators, max_depth, m1_name, m1_val, m2_name, m2_val
                )

                st.success("✅ Pelatihan selesai! Riwayat metrik otomatis tercatat di SQLite.")
                res_c1, res_c2 = st.columns(2)
                res_c1.metric(m1_name, f"{m1_val:.4f}")
                res_c2.metric(m2_name, f"{m2_val:.4f}")

                # Feature Importance
                st.subheader("Tingkat Kepentingan Fitur (Feature Importance)")
                fi_df = pd.DataFrame({
                    "Fitur": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Fitur"))

                # Tabel Hasil Prediksi Data Uji & Ekspor
                st.subheader("Hasil Prediksi pada Data Uji")
                df_pred = df_clean.loc[X_test.index].copy()
                df_pred[f"Prediksi_{target_col}"] = pred_output
                st.dataframe(df_pred.head(25), use_container_width=True)

                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    st.download_button(
                        "⬇️ Unduh Hasil Prediksi (CSV)",
                        data=df_pred.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_ml.csv",
                        mime="text/csv"
                    )
                with dl_c2:
                    st.download_button(
                        "⬇️ Unduh Hasil Prediksi (Excel)",
                        data=export_to_excel(df_pred),
                        file_name="hasil_prediksi_ml.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # ------------------------------------------
    # TAB 3: RIWAYAT DATABASE SQLITE
    # ------------------------------------------
    with tab3:
        st.subheader("Riwayat Eksperimen Model di SQLite (`ml_experiments.db`)")
        logs_df = get_experiment_logs()
        if logs_df.empty:
            st.info("Belum ada riwayat eksperimen. Silakan latih model di Tab 2 terlebih dahulu.")
        else:
            st.dataframe(logs_df, use_container_width=True)
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                st.download_button(
                    "⬇️ Ekspor Log SQLite ke Excel (.xlsx)",
                    data=export_to_excel(logs_df),
                    file_name="riwayat_eksperimen_sqlite.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col_db2:
                if st.button("🗑️ Bersihkan Seluruh Log SQLite"):
                    conn = sqlite3.connect(DB_NAME)
                    conn.execute("DELETE FROM experiment_logs")
                    conn.commit()
                    conn.close()
                    st.rerun()
else:
    st.info("Silakan unggah file CSV/Excel di panel sebelah kiri atau pilih Dataset Sintetis untuk memulai.")
