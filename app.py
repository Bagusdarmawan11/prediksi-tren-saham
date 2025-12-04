import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Stock Trend AI - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'Light'

# ==========================================
# 2. CSS CUSTOM
# ==========================================
if st.session_state.theme_mode == 'Dark':
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    chart_template = "plotly_dark"
else:
    bg_color = "#F0F2F6"
    text_color = "#31333F"
    card_bg = "#FFFFFF"
    chart_template = "plotly_white"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .metric-card {{
        background-color: {card_bg};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #ddd;
    }}
    .sidebar-logo-container {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        background-color: {card_bg};
        padding: 10px;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    # Cari file model secara fleksibel
    model_name = 'model_lstm_stock_trend.keras'
    scaler_name = 'scaler_stock_trend.pkl'
    params_name = 'model_params.pkl'
    
    # Cek path (Root atau folder main)
    path = ""
    if not os.path.exists(model_name):
        if os.path.exists(os.path.join("main", model_name)):
            path = "main/"
        else:
            return None, None, None # File tidak ditemukan

    try:
        model = tf.keras.models.load_model(path + model_name)
        with open(path + scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        with open(path + params_name, 'rb') as f:
            params = pickle.load(f)
        return model, scaler, params
    except Exception:
        return None, None, None

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def add_technical_indicators(df):
    df = df.copy()
    # Hitung indikator persis seperti notebook
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std_dev)
    
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    return df

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/9/98/Logo_Ubhara_Jaya.png"
    
    st.markdown(f"""
    <div class="sidebar-logo-container">
        <img src="{logo_url}" width="50">
        <div style="margin-left:10px;">
            <div style="font-weight:bold; font-size:14px; color:{text_color};">Bagus Darmawan</div>
            <div style="font-size:11px; color:gray;">NPM: 202210715059</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Pengaturan")
    theme = st.radio("Tema Aplikasi", ["Light", "Dark"], horizontal=True)
    if theme != st.session_state.theme_mode:
        st.session_state.theme_mode = theme
        st.rerun()
        
    st.divider()
    ticker = st.text_input("Kode Saham", value="BBCA.JK").upper()

# ==========================================
# 6. DISCLAIMER (VERSI STABIL)
# ==========================================
if not st.session_state.disclaimer_accepted:
    st.warning("⚠️ **DISCLAIMER (PENAFIAN)**")
    st.info("""
    1. Aplikasi ini adalah **proyek akademik** Universitas Bhayangkara Jakarta Raya.
    2. Hasil prediksi AI **tidak menjamin keuntungan** dan bisa salah.
    3. Gunakan sebagai referensi tambahan, bukan saran investasi mutlak.
    """)
    if st.button("Saya Mengerti & Lanjutkan"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

# ==========================================
# 7. MAIN CONTENT
# ==========================================

st.title("📈 Prediksi Tren Saham AI")
st.write(f"Analisis Pergerakan Saham: **{ticker}**")

model, scaler, params = load_resources()

if model is None:
    st.error("❌ Model tidak ditemukan. Pastikan file .keras dan .pkl sudah diupload ke GitHub.")
else:
    if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
        with st.spinner('Sedang memproses...'):
            try:
                df_raw = yf.download(ticker, period='2y', progress=False)
                if len(df_raw) > 60:
                    # Fix MultiIndex
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    
                    # Process
                    df_processed = add_technical_indicators(df_raw)
                    features = params.get('feature_columns', [])
                    look_back = params.get('look_back', 30)
                    available_cols = [c for c in features if c in df_processed.columns]
                    
                    # Predict
                    input_data = df_processed[available_cols].values[-look_back:]
                    input_scaled = scaler.transform(input_data)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))
                    
                    proba = model.predict(X_input)[0]
                    pred_class = np.argmax(proba)
                    
                    labels = ['DOWNTREND 📉', 'SIDEWAYS ➡️', 'UPTREND 🚀']
                    result = labels[pred_class]
                    confidence = proba[pred_class] * 100
                    
                    # Show Metrics
                    c1, c2, c3 = st.columns(3)
                    
                    # Tampilan Metrik HTML
                    metrics = [
                        ("Harga Terakhir", f"Rp {df_raw['Close'].iloc[-1]:,.0f}"),
                        ("Prediksi AI", result),
                        ("Keyakinan", f"{confidence:.2f}%")
                    ]
                    
                    for i, col in enumerate([c1, c2, c3]):
                        col.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:12px; color:gray;">{metrics[i][0]}</div>
                            <div style="font-size:20px; font-weight:bold; color:{text_color};">{metrics[i][1]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()
                    
                    # Grafik Harga
                    st.subheader("📊 Grafik Pergerakan Harga")
                    plot_data = df_processed.iloc[-100:]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], mode='lines', name='Close', line=dict(color='#2962FF', width=2)))
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Upper'], line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Lower'], fill='tonexty', fillcolor='rgba(41, 98, 255, 0.1)', line=dict(width=0), showlegend=False))
                    fig.update_layout(template=chart_template, height=450, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Tanggal", yaxis_title="Harga", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Grafik Probabilitas
                    st.subheader("🧠 Probabilitas")
                    prob_df = pd.DataFrame({"Kategori": ["Turun", "Datar", "Naik"], "Nilai": proba})
                    fig_bar = go.Figure(data=[go.Bar(x=prob_df['Kategori'], y=prob_df['Nilai'], marker_color=['#FF4B4B', '#808495', '#09AB3B'], text=(prob_df['Nilai']*100).map('{:.1f}%'.format), textposition='auto')])
                    fig_bar.update_layout(template=chart_template, height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(showgrid=False))
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                else:
                    st.warning("Data tidak cukup.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 8. FOOTER
# ==========================================
st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: black; text-align: center; padding: 10px; font-size: 12px; border-top: 1px solid #ddd; z-index: 100;">
    Copyright © <b>Bagus Darmawan</b> (202210715059) | Universitas Bhayangkara Jakarta Raya
</div>
<div style="margin-bottom: 50px;"></div>
""", unsafe_allow_html=True)
