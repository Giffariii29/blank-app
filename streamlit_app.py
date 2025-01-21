import streamlit as st
import pickle
import numpy as np
import json
import pandas as pd
from tensorflow.keras.models import load_model

# Load mapping dari encode_dict.json
with open('encode_dict.json', 'r') as file:
    encode_dict = json.load(file)

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load LSTM model
lstm_model = load_model('lstm_model.h5')

# Load SVM classifier
with open('svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Judul Streamlit
st.title('Prediksi Diagnosis Kanker Payudara')
st.markdown('Aplikasi ini memprediksi apakah kanker payudara *benign (jinak)* atau *malignant (ganas)* berdasarkan data diagnostik.')

# Tampilkan input dalam form
st.markdown("### Silakan masukkan data diagnostik di bawah ini:")
with st.form(key='diagnosis_form'):
    expected_features = scaler.feature_names_in_
    input_features = []

    # Loop untuk menampilkan input yang lebih deskriptif
    for feature in expected_features:
        user_input = st.number_input(
            f"{feature.replace('_', ' ').capitalize()}",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=f"Masukkan nilai diagnostik untuk fitur {feature.replace('_', ' ')}."
        )
        input_features.append(user_input)

    # Tombol untuk mengirimkan form
    submit_button = st.form_submit_button(label="Prediksi")

# Buat DataFrame jika tombol submit ditekan
if submit_button:
    df = pd.DataFrame([input_features], columns=expected_features)

    def prediction(df):
        # Validasi jumlah fitur input
        if df.shape[1] != len(expected_features):
            raise ValueError(f"Jumlah fitur input tidak sesuai. Diharapkan {len(expected_features)} fitur, tetapi menerima {df.shape[1]} fitur.")

        # Skala fitur numerik menggunakan scaler
        scaled_data = scaler.transform(df)

        # Bentuk ulang input untuk LSTM
        input_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

        # Ekstrak fitur menggunakan model LSTM
        features = lstm_model.predict(input_data)

        # Lakukan prediksi menggunakan model SVM
        prediction = svm_classifier.predict(features)
        if prediction[0] == 1:
            result = 'Malignant (Ganas)'
        else:
            result = 'Benign (Jinak)'
        return result

    # Prediksi dan tampilkan hasil
    try:
        prediction_result = prediction(df)
        st.success(f"Hasil prediksi: **{prediction_result}**")
    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
