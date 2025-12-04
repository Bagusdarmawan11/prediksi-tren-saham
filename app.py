# =====================================================
#   APP STOCK TREND PREDICTION - REFACTORED VERSION
#   Author: Bagus Darmawan
#   Framework: Streamlit
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# =====================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# =====================================================
st.set_page_config(
    page_title="Prediksi Tren Harga Saham - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state.setdefault("disclaimer_accepted", False)
st.session_state.setdefault("theme_mode", "Light")

# =====================================================
# 2. RESOURCE LOADER
# =====================================================
@st.cache_resource
def load_resources():
    """Load model, scaler, and parameter file."""
    try:
        model = tf.keras.models.load_model("model_lstm_stock_trend.keras")

        with open("scaler_stock_trend.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("model_params.pkl", "rb") as f:
            params = pickle.load(f)

        return model, scaler, params

    except Exception:
        return None, None, None


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
def add_technical_indicators(df: pd.DataFrame):
    """Add SMA, EMA, RSI, MACD, Bollinger Bands, Log Return."""
    df = df.copy()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    # Log return
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df.dropna()


# =====================================================
# 4. DISCLAIMER POP-UP
# =====================================================
@st.dialog("⚠️ DISCLAIMER (PENAFIAN)")
def show_disclaimer():
    st.write("""
    **Harap dibaca dengan seksama:**

    1. Aplikasi ini adalah proyek akademik untuk keperluan pembelajaran.
    2. Prediksi AI tidak menjamin akurasi 100%.
    3. Semua risiko kerugian akibat keputusan pengguna adalah tanggung jawab pengguna.
    4. Pasar saham memiliki risiko tinggi — lakukan riset mandiri (DYOR).
    """)

    if st.button("Saya Mengerti & Lanjutkan"):
        st.session_state.disclaimer_accepted = True
        st.rerun()


if not st.session_state.disclaimer_accepted:
    show_disclaimer()

# =====================================================
# 5. THEME COLORS
# =====================================================
THEMES = {
    "Dark":  {"bg": "#0E1117", "text": "#FAFAFA", "card": "#262730", "chart": "plotly_dark"},
    "Light": {"bg": "#F0F2F6", "text": "#31333F", "card": "#FFFFFF", "chart": "plotly_white"},
}

theme = THEMES[st.session_state.theme_mode]

# Inject CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    .metric-card {{
        background-color: {theme['card']};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .footer {{
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 8px;
        font-size: 12px;
        text-align: center;
        background-color: {theme['card']};
        border-top: 1px solid #ccc;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 6. SIDEBAR
# =====================================================
with st.sidebar:

    # Profile
    st.markdown("""
    <div style="display:flex;align-items:center;margin-bottom:15px;">
        <img src="https://upload.wikimedia.org/wikipedia/id/b/b4/Logo_ubhara.png" width="60">
        <div style="margin-left:10px;">
            <b>Bagus Darmawan</b><br>
            NPM: 202210715059<br>
            <span style="font-size:11px;">Universitas Bhayangkara Jakarta Raya</span>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.header("⚙️ Pengaturan")
    mode = st.radio("Tema Aplikasi", ["Light", "Dark"], horizontal=True)

    if mode != st.session_state.theme_mode:
        st.session_state.theme_mode = mode
        st.rerun()

    ticker = st.text_input("Kode Saham", "BBCA.JK").upper()
    st.info("Gunakan kode Yahoo Finance: BBRI.JK, TLKM.JK, GOTO.JK, dll.")

# =====================================================
# 7. MAIN CONTENT
# =====================================================
st.title("📈 Prediksi Tren Harga Saham Menggunakan LSTM/GRU")
st.write(f"Analisis prediksi tren untuk saham **{ticker}**")

model, scaler, params = load_resources()

if model is None:
    st.error("Model atau scaler tidak ditemukan!")
    st.stop()

# ----------------------------
# BUTTON: MULAI ANALISIS
# ----------------------------
if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):

    with st.spinner("Mengambil data & melakukan prediksi..."):

        try:
            df_raw = yf.download(ticker, period="1y", progress=False)

            if len(df_raw) < 60:
                st.warning("Data historis tidak cukup.")
                st.stop()

            # Fix multiindex (jika ada)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.droplevel(1)

            df = add_technical_indicators(df_raw)

            # Get model params
            features = params.get("feature_columns", [])
            look_back = params.get("look_back", 30)

            available_features = [c for c in features if c in df.columns]

            if len(available_features) < len(features):
                st.warning("Beberapa fitur tidak tersedia — akurasi mungkin menurun.")

            # Prepare input
            input_data = df[available_features].values[-look_back:]
            scaled = scaler.transform(input_data)
            X = scaled.reshape(1, look_back, len(available_features))

            # Prediction
            proba = model.predict(X)[0]
            pred = np.argmax(proba)

            labels = ["DOWNTREND 📉", "SIDEWAYS ➡️", "UPTREND 🚀"]
            result = labels[pred]
            confidence = proba[pred] * 100

            # -----------------------
            # METRIC CARDS
            # -----------------------
            st.subheader("📊 Hasil Prediksi")

            col1, col2, col3 = st.columns(3)
            metric_titles = ["Harga Terakhir", "Prediksi Tren", "Confidence"]
            metric_values = [
                f"Rp {df_raw['Close'].iloc[-1]:,.0f}",
                result,
                f"{confidence:.2f}%"
            ]

            for col, title, val in zip([col1, col2, col3], metric_titles, metric_values):
                col.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="color:gray;font-size:14px;">{title}</div>
                        <div style="font-size:24px;font-weight:bold;">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            # -----------------------
            # PRICE CHART
            # -----------------------
            st.subheader("📈 Grafik Harga & Indikator")

            plot_df = df.iloc[-100:]
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#2962FF", width=2)
            ))

            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["BB_Upper"],
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df["BB_Lower"],
                fill="tonexty",
                fillcolor="rgba(41,98,255,0.1)",
                line=dict(width=0),
                name="Bollinger Bands"
            ))

            fig.update_layout(
                template=theme["chart"],
                height=480,
                hovermode="x unified",
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------
            # BAR CHART PROBABILITY
            # -----------------------
            st.subheader("🧠 Probabilitas AI")

            bar_colors = ["#FF4B4B", "#808495", "#09AB3B"]
            cat = ["Downtrend", "Sideways", "Uptrend"]

            fig2 = go.Figure(go.Bar(
                x=cat,
                y=proba,
                marker_color=bar_colors,
                text=[f"{p*100:.1f}%" for p in proba],
                textposition="auto"
            ))

            fig2.update_layout(
                template=theme["chart"],
                height=300,
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# =====================================================
# 8. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    © <b>Bagus Darmawan</b> - NPM: 202210715059 | Universitas Bhayangkara Jakarta Raya
</div>
""", unsafe_allow_html=True)
