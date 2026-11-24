import streamlit as st

st.set_page_config(page_title="EEG Eye-State Monitoring", page_icon="🧠", layout="wide")

st.title("🧠 EEG Eye-State Intelligent Monitoring")
st.caption("Kelompok 14 · Dashboard Mata Kuliah Kecerdasan Komputasional")

st.markdown(
    """
**Pipeline:** Dataset UCI EEG Eye State → preprocessing → **GA** (seleksi channel)
→ **PSO** (hyperparameter LSTM) → **LSTM** → **Fuzzy Mamdani** (Awake / Transition / Drowsiness Indication).

> Sistem ini bukan alat diagnosis medis.

### Halaman
- **GA Optimization** (Minggu 13): seleksi channel EEG dengan Genetic Algorithm, kurva konvergensi,
  tabel fitur sebelum/sesudah, dan perbandingan Baseline vs GA+LSTM.

Pilih halaman di sidebar kiri.
"""
)
