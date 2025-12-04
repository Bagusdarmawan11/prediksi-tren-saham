import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="Prediksi Tren Harga Saham - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Session State untuk Disclaimer & Tema
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'Light'

# ==========================================
# 2. FUNGSI-FUNGSI UTILITAS
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

def add_technical_indicators(df):
    df = df.copy()
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
# 3. POP-UP DISCLAIMER (DIALOG)
# ==========================================
@st.dialog("⚠️ DISCLAIMER (PENAFIAN)")
def show_disclaimer():
    st.write("""
    **Harap dibaca dengan seksama:**
    
    1. Aplikasi ini merupakan **proyek akademik** untuk tujuan penelitian dan pembelajaran.
    2. Prediksi yang dihasilkan oleh AI (Artificial Intelligence) **tidak menjamin akurasi 100%** dan tidak boleh dijadikan satu-satunya landasan pengambilan keputusan investasi.
    3. Segala kerugian finansial yang timbul akibat penggunaan informasi dari aplikasi ini adalah **tanggung jawab pengguna sepenuhnya**.
    4. Pasar saham memiliki risiko tinggi. Selalu lakukan riset mandiri (DYOR).
    """)
    if st.button("Saya Mengerti & Lanjutkan"):
        st.session_state.disclaimer_accepted = True
        st.rerun()

if not st.session_state.disclaimer_accepted:
    show_disclaimer()

# ==========================================
# 4. CSS CUSTOM (DARK/LIGHT MODE & STYLING)
# ==========================================

# Warna berdasarkan Mode
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
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }}
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
        z-index: 100;
    }}
    /* Logo Styling in Sidebar */
    .sidebar-logo-container {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }}
    .sidebar-text {{
        margin-left: 15px;
        line-height: 1.2;
    }}
    .sidebar-name {{
        font-weight: bold;
        font-size: 16px;
    }}
    .sidebar-npm {{
        font-size: 12px;
        color: gray;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. SIDEBAR (PROFIL & SETTINGS)
# ==========================================
with st.sidebar:
    # Logo Ubhara & Profil
    # Gunakan URL logo Ubhara Jaya (pastikan link valid)
    logo_url = "https://upload.wikimedia.org/wikipedia/id/b/b4/Logo_ubhara.png" 
    
    st.markdown(f"""
    <div class="sidebar-logo-container">
        <img src="{logo_url}" width="60">
        <div class="sidebar-text">
            <div class="sidebar-name">Bagus Darmawan</div>
            <div class="sidebar-npm">NPM: 202210715059</div>
            <div style="font-size: 10px; margin-top:5px;">Universitas Bhayangkara<br>Jakarta Raya</div>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Pengaturan")
    
    # Dark/Light Mode Toggle
    mode = st.radio("Tema Aplikasi", ["Light", "Dark"], horizontal=True)
    if mode != st.session_state.theme_mode:
        st.session_state.theme_mode = mode
        st.rerun()

    st.subheader("Parameter Model")
    ticker = st.text_input("Kode Saham", value="BBCA.JK").upper()
    
    st.info("💡 **Tips:** Gunakan kode saham Yahoo Finance (Contoh: BBRI.JK, TLKM.JK, GOTO.JK)")

# ==========================================
# 6. MAIN CONTENT
# ==========================================

st.title("📈 Prediksi Tren Harga Saham Menggunakan LSTM dan GRU")
st.markdown(f"Analisis Pergerakan Saham **{ticker}** Menggunakan LSTM/GRU")

# Load Resources
model, scaler, params = load_resources()

if model is None:
    st.error("File Model/Scaler tidak ditemukan. Silakan upload file .keras dan .pkl.")
else:
    if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
        
        with st.spinner('Mengambil data pasar & melakukan prediksi...'):
            try:
                # 1. Get Data
                df_raw = yf.download(ticker, period='1y', progress=False)
                
                if len(df_raw) > 60:
                    # Flatten MultiIndex
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    
                    # Feature Engineering
                    df_processed = add_technical_indicators(df_raw)
                    
                    # Prepare Input
                    features = params.get('feature_columns', [])
                    look_back = params.get('look_back', 30)
                    
                    # Check columns
                    available_cols = [c for c in features if c in df_processed.columns]
                    if len(available_cols) != len(features):
                        st.warning("Fitur data tidak lengkap, akurasi mungkin berkurang.")
                    
                    input_data = df_processed[available_cols].values[-look_back:]
                    input_scaled = scaler.transform(input_data)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))
                    
                    # Predict
                    proba = model.predict(X_input)[0]
                    pred_class = np.argmax(proba)
                    
                    # Results
                    labels = ['DOWNTREND 📉', 'SIDEWAYS ➡️', 'UPTREND 🚀']
                    result_text = labels[pred_class]
                    confidence = proba[pred_class] * 100
                    
                    # --- TAMPILAN METRICS ---
                    st.markdown("### 📊 Hasil Prediksi")
                    col1, col2, col3 = st.columns(3)
                    
                    # Custom Metric Cards using HTML
                    cols = [col1, col2, col3]
                    titles = ["Harga Terakhir", "Prediksi Tren", "Keyakinan (Confidence)"]
                    values = [
                        f"Rp {df_raw['Close'].iloc[-1]:,.0f}", 
                        result_text, 
                        f"{confidence:.2f}%"
                    ]
                    
                    for i, c in enumerate(cols):
                        c.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:14px; color:gray;">{titles[i]}</div>
                            <div style="font-size:24px; font-weight:bold; color:{text_color};">{values[i]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    # --- GRAFIK HARGA (PLOTLY) ---
                    st.subheader("📈 Grafik Pergerakan Harga & Indikator")
                    
                    plot_data = df_processed.iloc[-100:]
                    
                    fig = go.Figure()
                    
                    # Close Price
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['Close'], 
                        mode='lines', name='Close Price', 
                        line=dict(color='#2962FF', width=2)
                    ))
                    
                    # Bollinger Bands Area
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Upper'], 
                        mode='lines', name='BB Upper', 
                        line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data.index, y=plot_data['BB_Lower'], 
                        mode='lines', name='Bollinger Bands', 
                        line=dict(width=0), fill='tonexty', 
                        fillcolor='rgba(41, 98, 255, 0.1)'
                    ))

                    fig.update_layout(
                        template=chart_template,
                        height=500,
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (IDR)",
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- GRAFIK PROBABILITAS (WARNA DIPERBAIKI) ---
                    st.subheader("🧠 Detail Probabilitas AI")
                    
                    probs_df = pd.DataFrame({
                        "Kategori": ["Downtrend", "Sideways", "Uptrend"],
                        "Probabilitas": proba
                    })
                    
                    # Warna khusus: Merah (Turun), Abu (Datar), Hijau (Naik)
                    colors = ['#FF4B4B', '#808495', '#09AB3B']
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=probs_df['Kategori'],
                        y=probs_df['Probabilitas'],
                        marker_color=colors,
                        text=(probs_df['Probabilitas']*100).map('{:.1f}%'.format),
                        textposition='auto'
                    )])
                    
                    fig_bar.update_layout(
                        template=chart_template,
                        height=300,
                        yaxis=dict(range=[0, 1], showgrid=False),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                else:
                    st.warning("Data historis saham tidak cukup untuk melakukan analisis.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

# ==========================================
# 7. FOOTER COPYRIGHT
# ==========================================
st.markdown(f"""
<div class="footer">
    Copyright © <b>Bagus Darmawan</b> - NPM: <b>202210715059</b> | Universitas Bhayangkara Jakarta Raya
</div>
""", unsafe_allow_html=True)
