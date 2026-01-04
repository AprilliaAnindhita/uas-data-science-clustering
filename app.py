import streamlit as st
import numpy as np
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
pipeline = joblib.load("clustering_pipeline.pkl")

# =========================
# LABEL CLUSTER (3 CLUSTER)
# =========================
cluster_labels = {
    0: {
        "title": "Nasabah Aktivitas Rata-Rata & Saldo Rendah",
        "desc": "Nasabah dengan transaksi rutin, saldo relatif rendah, dan aktivitas login minimal."
    },
    1: {
        "title": "Nasabah Mapan & Aktif",
        "desc": "Nasabah produktif dengan saldo lebih besar dan aktivitas transaksi stabil."
    },
    2: {
        "title": "Nasabah Muda Aktif & Digital",
        "desc": "Nasabah relatif muda dengan pola transaksi stabil dan potensi adopsi layanan digital."
    }
}

cluster_colors = {
    0: "#FDE68A",  # kuning
    1: "#BFDBFE",  # biru
    2: "#BBF7D0"   # hijau
}

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>📊 Customer Segmentation Dashboard</h1>
<p style='text-align:center; font-size:16px;'>
Aplikasi untuk memprediksi segmentasi nasabah menggunakan <b>K-Means Clustering</b>
</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🧾 Input Data Nasabah")

transaction_amount = st.sidebar.number_input(
    "Transaction Amount", min_value=0.0, value=250.0
)
customer_age = st.sidebar.number_input(
    "Customer Age", min_value=18, max_value=100, value=40
)
transaction_duration = st.sidebar.number_input(
    "Transaction Duration (detik)", min_value=0, value=120
)
login_attempts = st.sidebar.number_input(
    "Login Attempts", min_value=0, value=1
)
account_balance = st.sidebar.number_input(
    "Account Balance", min_value=0.0, value=5000000.0
)

# =========================
# TOMBOL PREDIKSI
# =========================
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Prediksi Cluster", use_container_width=True)

# =========================
# PROSES PREDIKSI
# =========================
if predict_btn:

    # Susun 13 fitur sesuai urutan training
    input_data = np.array([[
        transaction_amount,   # TransactionAmount
        0,                     # PreviousTransactionDate (dummy)
        0,                     # TransactionType (dummy)
        0,                     # Location (dummy)
        0,                     # Channel (dummy)
        customer_age,          # CustomerAge
        0,                     # CustomerOccupation (dummy)
        transaction_duration,  # TransactionDuration
        login_attempts,        # LoginAttempts
        account_balance,       # AccountBalance
        0,                     # TransactionDate (dummy)
        0,                     # Amount_Binned (dummy)
        0                      # Age_Binned (dummy)
    ]])

    cluster_result = pipeline.predict(input_data)
    cluster_id = int(cluster_result[0])
    cluster_info = cluster_labels[cluster_id]
    color = cluster_colors[cluster_id]

    # =========================
    # OUTPUT HASIL
    # =========================
    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:30px;
        border-radius:15px;
        text-align:center;
        box-shadow:0 4px 10px rgba(0,0,0,0.1);
        margin-top:20px;
    ">
        <h2>Cluster {cluster_id}</h2>
        <h3>{cluster_info['title']}</h3>
        <p style="font-size:16px;">{cluster_info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

