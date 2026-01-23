import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# CEK DATA PREPROCESSING
# =========================
if 'tweet_df1' not in st.session_state:
    st.warning("Data belum tersedia. Jalankan preprocessing & labeling dulu.")
    st.stop()

df = st.session_state.tweet_df1

X = np.array(df['padded_sequence']) if 'padded_sequence' in df else None
y = np.array(df['sentiment_vader_2Sentiment'].factorize()[0])

# =========================
# AMBIL LIST MODEL .h5
# =========================
MODEL_DIR = "save_modelling"

if not os.path.exists(MODEL_DIR):
    st.error("Folder save_modelling tidak ditemukan.")
    st.stop()

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

if not model_files:
    st.warning("Belum ada model .h5 yang tersimpan.")
    st.stop()

# =========================
# PILIH MODEL
# =========================
st.subheader("Pilih Model")
selected_model = st.selectbox(
    "Model tersedia:",
    model_files
)

# =========================
# LOAD & EVALUATE
# =========================
if st.button("Load & Evaluasi Model"):
    model_path = os.path.join(MODEL_DIR, selected_model)

    with st.spinner("Loading model..."):
        model = load_model(model_path)

    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5)

    acc = accuracy_score(y, y_pred) * 100
    prec = precision_score(y, y_pred, average='weighted') * 100
    rec = recall_score(y, y_pred, average='weighted') * 100
    f1 = f1_score(y, y_pred, average='weighted') * 100

    st.success("Model berhasil dievaluasi 🎉")

    col1, col2 = st.columns(2)
    col1.metric("Akurasi", f"{acc:.2f}%")
    col2.metric("Precision", f"{prec:.2f}%")

    col3, col4 = st.columns(2)
    col3.metric("Recall", f"{rec:.2f}%")
    col4.metric("F1-Score", f"{f1:.2f}%")
