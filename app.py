import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pickle
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Stock Trend AI - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. FUNGSI LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('model_lstm_stock_trend.keras')
        with open('scaler_stock_trend.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_params.pkl', 'rb') as f:
            params = pickle.load(f)
        return model, scaler, params
    except Exception as e:
        return None, None, None

# ==========================================
# 3. FUNGSI FEATURE ENGINEERING
# ==========================================
def add_technical_indicators(df):
    df = df.copy()
    # Hitung indikator sama persis seperti saat training
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
# 4. SIDEBAR (Profil & Logo)
# ==========================================
with st.sidebar:
    # Logo Universitas (Menggunakan HTML agar rapi)
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/98/Logo_Ubhara_Jaya.png" width="60" style="margin-right: 15px;">
        <div>
            <h3 style="margin: 0; font-size: 16px;">Bagus Darmawan</h3>
            <p style="margin: 0; font-size: 12px; color: gray;">NPM: 202210715059</p>
            <p style="margin: 0; font-size: 10px;">Univ. Bhayangkara Jakarta Raya</p>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Input Saham")
    ticker = st.text_input("Kode Ticker", value="BBCA.JK").upper()
    st.caption("Contoh: BBRI.JK, TLKM.JK, ASII.JK")

# ==========================================
# 5. HALAMAN UTAMA
# ==========================================

# --- DISCLAIMER (Gunakan st.warning/expander agar lebih stabil) ---
with st.expander("⚠️ **DISCLAIMER (PENAFIAN) - Harap Baca Terlebih Dahulu**", expanded=True):
    st.warning("""
    1. Aplikasi ini adalah **proyek akademik** untuk penelitian.
    2. Hasil prediksi AI **tidak menjamin akurasi 100%**.
    3. Segala keputusan investasi dan risikonya ada di tangan pengguna.
    """)

st.title("📈 Analisis Tren Saham AI")
st.write(f"Melakukan forecasting tren untuk saham: **{ticker}**")

# Load Model
model, scaler, params = load_resources()

if model is None:
    st.error("⚠️ File Model belum terdeteksi. Pastikan file .keras, .pkl ada di GitHub.")
else:
    # Tombol Prediksi
    if st.button("Mulai Analisis", type="primary", use_container_width=True):
        
        with st.spinner('Sedang memproses data...'):
            try:
                # Ambil Data
                df_raw = yf.download(ticker, period='1y', progress=False)
                
                if len(df_raw) > 60:
                    # Fix MultiIndex column issue from yfinance
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    
                    # Feature Engineering
                    df_processed = add_technical_indicators(df_raw)
                    
                    # Cek Kelengkapan Fitur
                    features = params.get('feature_columns', [])
                    look_back = params.get('look_back', 30)
                    
                    available_cols = [c for c in features if c in df_processed.columns]
                    
                    # Prepare Data
                    input_data = df_processed[available_cols].values[-look_back:]
                    input_scaled = scaler.transform(input_data)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))
                    
                    # Prediksi
                    proba = model.predict(X_input)[0]
                    pred_class = np.argmax(proba)
                    
                    # Mapping Label
                    labels = ['DOWNTREND 📉', 'SIDEWAYS ➡️', 'UPTREND 🚀']
                    result_text = labels[pred_class]
                    confidence = proba[pred_class] * 100
                    
                    # --- TAMPILAN HASIL ---
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Harga Terakhir", f"Rp {df_raw['Close'].iloc[-1]:,.0f}")
                    c2.metric("Prediksi AI", result_text)
                    c3.metric("Confidence", f"{confidence:.2f}%")
                    
                    # --- GRAFIK PLOTLY INTERAKTIF ---
                    st.subheader("📊 Visualisasi Data")
                    
                    plot_data = df_processed.iloc[-120:] # Ambil 120 hari terakhir
                    
                    fig = go.Figure()
                    
                    # Garis Harga
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['Close'],
                        mode='lines', name='Harga Close',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Bollinger Bands
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Upper'],
                        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Lower'],
                        mode='lines', name='Bollinger Bands',
                        fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)',
                        line=dict(width=0), hoverinfo='skip'
                    ))

                    fig.update_layout(
                        title=f"Pergerakan Harga {ticker}",
                        yaxis_title="Harga (IDR)",
                        xaxis_title="Tanggal",
                        template="plotly_white",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Grafik Probabilitas
                    st.write("### Probabilitas Prediksi")
                    prob_df = pd.DataFrame({
                        "Kategori": ["Turun (Down)", "Datar (Side)", "Naik (Up)"],
                        "Nilai": proba
                    })
                    
                    # Warna bar chart: Merah, Abu, Hijau
                    colors = ['#ff4b4b', '#808495', '#21c354']
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=prob_df['Kategori'],
                        y=prob_df['Nilai'],
                        marker_color=colors,
                        text=prob_df['Nilai'].apply(lambda x: f"{x*100:.1f}%"),
                        textposition='auto'
                    )])
                    fig_bar.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.warning("Data tidak cukup untuk melakukan prediksi.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: black; text-align: center; padding: 10px; border-top: 1px solid #ddd; z-index: 1000;">
    Copyright © <b>Bagus Darmawan</b> (202210715059) | Universitas Bhayangkara Jakarta Raya
</div>
<div style="margin-bottom: 60px;"></div>
""", unsafe_allow_html=True)
