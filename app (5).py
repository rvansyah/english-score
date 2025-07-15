import streamlit as st
import pandas as pd
import pickle

# --- Load Model ---
# Ganti 'model_nb.pkl' dengan path model kamu di repo
with open('model_nb.pkl', 'rb') as f:
    nb = pickle.load(f)

st.title("Prediksi Tingkat Kemampuan Bahasa Inggris")

st.write("Masukkan data baru untuk memprediksi tingkat kemampuan bahasa Inggris:")

# Input Form
with st.form("input_form"):
    new_absence_days = st.number_input("Jumlah absen harian:", min_value=0.0, step=1.0)
    new_weekly_self_study_hours = st.number_input("Jumlah jam belajar mingguan:", min_value=0.0, step=1.0)
    submit = st.form_submit_button("Prediksi")

if submit:
    # Buat DataFrame dari input baru
    new_data_df = pd.DataFrame(
        [[new_absence_days, new_weekly_self_study_hours]],
        columns=['absence_days','weekly_self_study_hours']
    )

    # Lakukan prediksi
    try:
        predicted_code = nb.predict(new_data_df)[0]  # hasilnya 0 atau 1 atau 2

        # Konversi hasil prediksi ke label asli
        label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"Prediksi tingkat kemampuan bahasa Inggris adalah: {predicted_label}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")