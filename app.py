import streamlit as st
import pandas as pd
import joblib  # untuk load model .pkl

# Load model yang sudah disimpan
nb = joblib.load('english_score.pkl')

# Mapping label
label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}

# Judul
st.title("Prediksi Tingkat Kemampuan Bahasa Inggris Siswa")

# Input dari user
absence_days = st.number_input("Masukkan jumlah absen harian", min_value=0.0, format="%.1f")
weekly_self_study_hours = st.number_input("Masukkan jumlah jam belajar mandiri per minggu", min_value=0.0, format="%.1f")

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Buat DataFrame dari input user
        new_data_df = pd.DataFrame([[absence_days, weekly_self_study_hours]],
                                   columns=['absence_days', 'weekly_self_study_hours'])

        # Prediksi
        predicted_code = nb.predict(new_data_df)[0]
        predicted_label = label_mapping.get(predicted_code, "Tidak diketahui")

        # Tampilkan hasil
        st.success(f"Prediksi tingkat kemampuan bahasa Inggris: **{predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
