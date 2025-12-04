import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Stock Trend AI - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State untuk Disclaimer & Tema
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'Light'

# ==========================================
# 2. CSS CUSTOM (UI PERMINTAAN ANDA)
# ==========================================
# Mengatur warna berdasarkan mode
if st.session_state.theme_mode == 'Dark':
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    chart_template = "plotly_dark"
    metric_color = "#E0E0E0"
else:
    bg_color = "#F0F2F6"
    text_color = "#31333F"
    card_bg = "#FFFFFF"
    chart_template = "plotly_white"
    metric_color = "#000000"

st.markdown(f"""
<style>
    /* Tema Global */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Kartu Metrik Custom */
    .metric-card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }}
    .metric-title {{
        font-size: 14px;
        color: gray;
        margin-bottom: 5px;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: {metric_color};
    }}
    
    /* Footer */
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {card_bg};
        color: {text_color};
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #ccc;
        z-index: 999;
    }}
    
    /* Sidebar Profil */
    .sidebar-container {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        background-color: {card_bg};
        padding: 10px;
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SMART FILE LOADER (SOLUSI ERROR FILE)
# ==========================================
@st.cache_resource
def load_resources():
    # Nama file yang dicari
    model_name = 'model_lstm_stock_trend.keras'
    scaler_name = 'scaler_stock_trend.pkl'
    params_name = 'model_params.pkl'

    # Cek lokasi file (Root atau dalam folder)
    current_dir = os.getcwd()
    files = os.listdir(current_dir)
    
    # Logika pencarian sederhana
    if model_name in files:
        path_prefix = ""
    elif "main" in files and os.path.exists(os.path.join("main", model_name)):
        path_prefix = "main/"
    else:
        # Jika tidak ketemu, return None dan list file untuk debugging
        return None, None, None, files

    try:
        model = tf.keras.models.load_model(path_prefix + model_name)
        with open(path_prefix + scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        with open(path_prefix + params_name, 'rb') as f:
            params = pickle.load(f)
        return model, scaler, params, []
    except Exception as e:
        return None, None, None, str(e)

# ==========================================
# 4. FITUR INDIKATOR TEKNIKAL
# ==========================================
def add_technical_indicators(df):
    df = df.copy()
    # SMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    # EMA
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    # BB
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std_dev)
    # Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df.dropna()

# ==========================================
# 5. SIDEBAR & NAVIGASI
# ==========================================
with st.sidebar:
    # Logo Universitas Bhayangkara Jakarta Raya
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/9/98/Logo_Ubhara_Jaya.png"
    
    st.markdown(f"""
    <div class="sidebar-container">
        <img src="{logo_url}" width="50">
        <div style="margin-left: 10px;">
            <div style="font-weight:bold; font-size:14px;">Bagus Darmawan</div>
            <div style="font-size:11px; color:gray;">NPM: 202210715059</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Pengaturan")
    
    # Mode Gelap/Terang
    new_theme = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

    st.divider()
    
    # Input Ticker
    st.subheader("Parameter Saham")
    ticker = st.text_input("Kode Saham", value="BBCA.JK").upper()
    st.caption("Contoh: BBRI.JK, TLKM.JK, ADRO.JK")

# ==========================================
# 6. POP-UP DISCLAIMER (Wajib Ada)
# ==========================================
if not st.session_state.disclaimer_accepted:
    with st.container():
        st.warning("⚠️ **DISCLAIMER (PENAFIAN) - DISCLAIMER ON!**")
        st.markdown("""
        1. Aplikasi ini adalah **proyek akademik** (Universitas Bhayangkara Jakarta Raya).
        2. Hasil prediksi AI **tidak menjamin akurasi 100%**.
        3. Pengembang tidak bertanggung jawab atas kerugian investasi.
        """)
        if st.button("Saya Mengerti & Lanjutkan"):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop() # Stop eksekusi kode di bawah sampai tombol ditekan

# ==========================================
# 7. HALAMAN UTAMA
# ==========================================

st.title(f"📈 Analisis Saham: {ticker}")
st.write("Prediksi tren menggunakan Deep Learning (LSTM/GRU).")

# Load Model
model, scaler, params, debug_info = load_resources()

if model is None:
    st.error("❌ **FILE MODEL TIDAK DITEMUKAN!**")
    st.info("Sistem mencari file di folder server, tapi tidak ketemu.")
    st.code(f"List file di server saat ini: {debug_info}")
    st.warning("Solusi: Pastikan file 'model_lstm_stock_trend.keras' ada di root GitHub Anda.")
else:
    # Tombol Analisis
    if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
        
        with st.spinner('Sedang mengambil data dan menghitung...'):
            try:
                # 1. Ambil Data
                df_raw = yf.download(ticker, period='2y', progress=False)
                
                if len(df_raw) > 60:
                    # Fix MultiIndex
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    
                    # Feature Engineering
                    df_processed = add_technical_indicators(df_raw)
                    
                    # Cek Fitur
                    features = params.get('feature_columns', [])
                    look_back = params.get('look_back', 30)
                    
                    available_cols = [c for c in features if c in df_processed.columns]
                    
                    # Prepare Input
                    input_data = df_processed[available_cols].values[-look_back:]
                    input_scaled = scaler.transform(input_data)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))
                    
                    # Predict
                    proba = model.predict(X_input)[0]
                    pred_class = np.argmax(proba)
                    
                    # Labels
                    labels = ['DOWNTREND 📉', 'SIDEWAYS ➡️', 'UPTREND 🚀']
                    result_text = labels[pred_class]
                    confidence = proba[pred_class] * 100
                    
                    # --- TAMPILAN METRIK KEREN ---
                    st.markdown("### 📊 Ringkasan Prediksi")
                    c1, c2, c3 = st.columns(3)
                    
                    cols = [c1, c2, c3]
                    titles = ["Harga Terakhir", "Prediksi AI", "Keyakinan"]
                    values = [f"Rp {df_raw['Close'].iloc[-1]:,.0f}", result_text, f"{confidence:.2f}%"]
                    
                    for i in range(3):
                        cols[i].markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">{titles[i]}</div>
                            <div class="metric-value">{values[i]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()

                    # --- GRAFIK HARGA (DIPERBAIKI) ---
                    st.subheader("📈 Grafik Harga & Bollinger Bands")
                    plot_data = df_processed.iloc[-150:]
                    
                    fig = go.Figure()
                    
                    # Close Price
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['Close'],
                        mode='lines', name='Harga Close',
                        line=dict(color='#2962FF', width=2)
                    ))
                    
                    # BB Upper & Lower
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Upper'],
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Lower'],
                        mode='lines', name='Bollinger Bands',
                        fill='tonexty', fillcolor='rgba(41, 98, 255, 0.15)',
                        line=dict(width=0)
                    ))

                    fig.update_layout(
                        template=chart_template,
                        height=450,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (IDR)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- GRAFIK PROBABILITAS ---
                    st.subheader("🧠 Probabilitas Model")
                    prob_df = pd.DataFrame({
                        "Kategori": ["Turun", "Datar", "Naik"],
                        "Nilai": proba
                    })
                    
                    # Warna: Merah, Abu, Hijau
                    bar_colors = ['#FF5252', '#9E9E9E', '#00C853']
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=prob_df['Kategori'],
                        y=prob_df['Nilai'],
                        marker_color=bar_colors,
                        text=(prob_df['Nilai']*100).map('{:.1f}%'.format),
                        textposition='auto'
                    )])
                    
                    fig_bar.update_layout(
                        template=chart_template,
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        yaxis=dict(range=[0, 1], showgrid=False)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                else:
                    st.warning("Data saham belum cukup untuk analisis AI.")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")

# ==========================================
# 8. COPYRIGHT FOOTER
# ==========================================
st.markdown(f"""
<div class="footer">
    Copyright © 2024 <b>Bagus Darmawan</b> - NPM: <b>202210715059</b> | Universitas Bhayangkara Jakarta Raya
</div>
<div style="margin-bottom: 50px;"></div>
""", unsafe_allow_html=True)
