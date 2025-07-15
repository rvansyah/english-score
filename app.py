import streamlit as st
import pandas as pd
import pickle
import os

st.title("Prediksi Tingkat Kemampuan Bahasa Inggris")

MODEL_PATH = 'model_nb.pkl'

def load_model(path):
    if not os.path.exists(path):
        st.error(f"File model '{path}' tidak ditemukan. Silakan pastikan file sudah di-upload ke repo.")
        return None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

nb = load_model(MODEL_PATH)

if nb:
    st.write("Masukkan data baru untuk memprediksi tingkat kemampuan bahasa Inggris:")
    with st.form("input_form"):
        new_absence_days = st.number_input("Jumlah absen harian:", min_value=0.0, step=1.0)
        new_weekly_self_study_hours = st.number_input("Jumlah jam belajar mingguan:", min_value=0.0, step=1.0)
        submit = st.form_submit_button("Prediksi")

    if submit:
        new_data_df = pd.DataFrame(
            [[new_absence_days, new_weekly_self_study_hours]],
            columns=['absence_days','weekly_self_study_hours']
        )
        try:
            predicted_code = nb.predict(new_data_df)[0]
            label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')
            st.success(f"Prediksi tingkat kemampuan bahasa Inggris adalah: {predicted_label}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
else:
    st.warning("Aplikasi belum bisa digunakan karena model belum tersedia atau gagal diload.")
