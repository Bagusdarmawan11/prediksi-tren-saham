import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="AI Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Tampilan Keren & Footer Copyright
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOGIKA & MODEL
# ==========================================

@st.cache_resource
def load_resources():
    """Load Model dan Scaler agar tidak berulang-ulang"""
    try:
        # Ganti dengan nama file model Anda yang benar
        model = tf.keras.models.load_model('model_lstm_stock_trend.keras')
        
        with open('scaler_stock_trend.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            
        return model, scaler, params
    except Exception as e:
        return None, None, None

def add_technical_indicators(df):
    """
    REPLIKASI FEATURE ENGINEERING DARI NOTEBOOK
    Wajib sama persis agar model tidak error dimensi.
    """
    df = df.copy()
    
    # 1. SMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # 2. EMA
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # 3. RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 5. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std_dev)

    # 6. Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Hapus NaN akibat rolling
    df = df.dropna()
    
    return df

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=100)
    st.title("Pengaturan")
    ticker = st.text_input("Kode Saham (Yahoo Finance)", value="BBCA.JK").upper()
    
    st.info("""
    **Panduan Label:**
    - 📈 **Uptrend:** Potensi Naik
    - 📉 **Downtrend:** Potensi Turun
    - ➡️ **Sideways:** Pasar Datar
    """)

# --- MAIN CONTENT ---
st.title("🤖 Prediksi Tren Saham dengan AI")
st.markdown(f"Analisis Real-time untuk saham: **{ticker}**")

# Load Model
model, scaler, params = load_resources()

if model is None:
    st.error("❌ File Model atau Scaler tidak ditemukan! Pastikan file .keras dan .pkl sudah diupload.")
else:
    # Button Prediksi
    if st.button("🔍 Analisis Sekarang", use_container_width=True):
        
        with st.spinner('Sedang mengunduh data pasar dan menghitung indikator...'):
            # 1. Ambil Data Lebih Banyak (untuk buffer indikator)
            df_raw = yf.download(ticker, period='1y', progress=False)
            
            if len(df_raw) > 50:
                # 2. Proses Data (Sama seperti notebook)
                try:
                    # Flatten MultiIndex columns jika ada (fix yfinance bug)
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)
                    
                    # Hitung Indikator
                    df_processed = add_technical_indicators(df_raw)
                    
                    # Cek Feature Columns dari params
                    features = params.get('feature_columns', [])
                    
                    # Validasi Kolom
                    available_cols = [c for c in features if c in df_processed.columns]
                    
                    if len(available_cols) != len(features):
                         st.warning("Beberapa fitur hilang, hasil mungkin kurang akurat.")
                    
                    # Ambil data terakhir sesuai LOOK_BACK
                    look_back = params.get('look_back', 30)
                    input_data = df_processed[available_cols].values[-look_back:]
                    
                    # Scaling
                    input_scaled = scaler.transform(input_data)
                    
                    # Reshape untuk LSTM (1, 30, 16)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))
                    
                    # 3. Prediksi
                    proba = model.predict(X_input)[0]
                    pred_class = np.argmax(proba)
                    
                    # Mapping Label
                    labels = {0: 'DOWNTREND 📉', 1: 'SIDEWAYS ➡️', 2: 'UPTREND 🚀'}
                    colors = {0: 'red', 1: 'gray', 2: 'green'}
                    
                    result = labels[pred_class]
                    confidence = proba[pred_class] * 100
                    
                    # 4. Tampilan Hasil (Dashboard)
                    st.divider()
                    
                    # Metrics Row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Harga Terakhir", f"Rp {df_raw['Close'].iloc[-1]:,.0f}")
                    with col2:
                        st.metric("Prediksi AI", result)
                    with col3:
                        st.metric("Keyakinan (Confidence)", f"{confidence:.2f}%")
                    
                    # 5. Grafik Interaktif Plotly
                    st.subheader("📊 Grafik Pergerakan Harga & Bollinger Bands")
                    
                    # Ambil data 3 bulan terakhir untuk plot
                    plot_data = df_processed.iloc[-90:]
                    
                    fig = go.Figure()
                    
                    # Candlestick (jika ada data OHLC) atau Line
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], 
                                            mode='lines', name='Close Price', line=dict(color='blue')))
                    
                    # Bollinger Bands
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Upper'], 
                                            mode='lines', name='BB Upper', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_Lower'], 
                                            mode='lines', name='BB Lower', line=dict(width=0), fill='tonexty', 
                                            fillcolor='rgba(173, 216, 230, 0.2)', showlegend=False))

                    fig.update_layout(
                        title=f"Analisis Teknikal {ticker}",
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (IDR)",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 6. Detail Probabilitas
                    st.subheader("🧠 Detail Analisis Probabilitas")
                    prob_df = pd.DataFrame({
                        "Kondisi": ["Downtrend", "Sideways", "Uptrend"],
                        "Probabilitas": proba
                    })
                    st.bar_chart(prob_df.set_index("Kondisi"), color="#2980b9")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {e}")
                    st.error("Pastikan model dilatih dengan fitur yang sama dengan yang dihitung di sini.")
            else:
                st.error("Data historis tidak ditemukan atau terlalu sedikit untuk analisis.")

# ==========================================
# 4. COPYRIGHT FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <p>Copyright © 2024 AI Stock Predictor | Developed with ❤️ using TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
