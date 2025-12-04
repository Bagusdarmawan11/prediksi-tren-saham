# =====================================================
#   APP STOCK TREND PREDICTION
#   Author : Bagus Darmawan
#   Description : Prediksi tren harga saham (LSTM/GRU)
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
# 1. PAGE CONFIG & SESSION STATE
# =====================================================
st.set_page_config(
    page_title="Prediksi Tren Harga Saham - Bagus Darmawan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

# =====================================================
# 2. THEME & GLOBAL STYLES
# =====================================================
THEMES = {
    "Dark": {
        "bg": "#0E1117",
        "text": "#FAFAFA",
        "card": "#1E2128",
        "sidebar": "#111318",
        "chart": "plotly_dark",
    },
    "Light": {
        "bg": "#F3F4F6",
        "text": "#111827",
        "card": "#FFFFFF",
        "sidebar": "#111318",  # sidebar tetap gelap biar kontras
        "chart": "plotly_white",
    },
}

current_theme = THEMES[st.session_state.theme_mode]
bg_color = current_theme["bg"]
text_color = current_theme["text"]
card_bg = current_theme["card"]
sidebar_bg = current_theme["sidebar"]
chart_template = current_theme["chart"]

# CSS global
st.markdown(
    f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        border-right: 1px solid #374151;
    }}

    .sidebar-profile {{
        display: flex;
        align-items: center;
        margin-bottom: 22px;
    }}

    .sidebar-profile img {{
        border-radius: 8px;
        border: 1px solid #4B5563;
    }}

    .sidebar-text {{
        margin-left: 14px;
        color: #E5E7EB;
        line-height: 1.3;
    }}

    .sidebar-name {{
        font-weight: 700;
        font-size: 15px;
    }}

    .sidebar-npm {{
        font-size: 12px;
        color: #9CA3AF;
    }}

    /* Header utama */
    .header-card {{
        background-color: {card_bg};
        padding: 32px 40px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin-bottom: 24px;
    }}

    .app-title {{
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
    }}

    .app-subtitle {{
        font-size: 0.95rem;
        color: #6B7280;
        margin-top: 4px;
    }}

    /* Metric cards */
    .metric-card {{
        background-color: {card_bg};
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.06);
        text-align: center;
    }}

    /* Primary button */
    .stButton>button {{
        border-radius: 999px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg,#f97373,#ef4444);
        color: white;
        box-shadow: 0 8px 18px rgba(248,113,113,0.45);
    }}

    .stButton>button:hover {{
        filter: brightness(1.03);
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(248,113,113,0.6);
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
        padding: 8px;
        font-size: 11px;
        border-top: 1px solid #E5E7EB;
        z-index: 100;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# 3. LOAD TICKERS FROM CSV
# =====================================================
@st.cache_resource
def load_tickers_from_csv(csv_path: str = "DaftarSaham.csv"):
    """
    Membaca file DaftarSaham.csv dan mengembalikan list kode
    dalam format 'XXXX.JK'.
    """
    try:
        df_codes = pd.read_csv(csv_path)

        codes = (
            df_codes["Code"]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
        )

        tickers = sorted([f"{code}.JK" for code in codes if code])
        return tickers

    except Exception as e:
        # fallback jika file tidak terbaca
        st.sidebar.warning(
            f"Gagal membaca DaftarSaham.csv: {e}. "
            "Menggunakan contoh beberapa ticker default."
        )
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "GOTO.JK"]


ID_TICKERS = load_tickers_from_csv()

# =====================================================
# 4. LOAD MODEL / RESOURCES
# =====================================================
@st.cache_resource
def load_resources():
    """Load model, scaler, dan parameter model."""
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
# 5. FEATURE ENGINEERING
# =====================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah SMA, EMA, RSI, MACD, Bollinger Bands, dan log return."""
    df = df.copy()

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    mid = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + (2 * std)
    df["BB_Lower"] = mid - (2 * std)

    # Log return
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df.dropna()


# =====================================================
# 6. DISCLAIMER
# =====================================================
@st.dialog("⚠️ DISCLAIMER (PENAFIAN)")
def show_disclaimer():
    st.write(
        """
**Harap dibaca dengan seksama:**

1. Aplikasi ini merupakan **proyek akademik** untuk tujuan penelitian dan pembelajaran.
2. Prediksi yang dihasilkan oleh AI **tidak menjamin akurasi 100%** dan tidak boleh dijadikan satu-satunya dasar keputusan investasi.
3. Segala kerugian finansial yang timbul akibat penggunaan informasi dari aplikasi ini adalah **tanggung jawab pengguna**.
4. Pasar saham memiliki risiko tinggi. Selalu lakukan riset mandiri (DYOR).
        """
    )
    if st.button("Saya Mengerti & Lanjutkan"):
        st.session_state.disclaimer_accepted = True
        st.rerun()


if not st.session_state.disclaimer_accepted:
    show_disclaimer()

# =====================================================
# 7. SIDEBAR
# =====================================================
with st.sidebar:
    # Profil
    st.markdown(
        """
        <div class="sidebar-profile">
            <img src="https://upload.wikimedia.org/wikipedia/id/b/b4/Logo_ubhara.png" width="60">
            <div class="sidebar-text">
                <div class="sidebar-name">Bagus Darmawan</div>
                <div class="sidebar-npm">NPM: 202210715059</div>
                <div style="font-size:11px;margin-top:3px;">
                    Universitas Bhayangkara<br>Jakarta Raya
                </div>
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    st.header("⚙️ Pengaturan")

    # Tema
    mode = st.radio("Tema Aplikasi", ["Light", "Dark"], horizontal=True)
    if mode != st.session_state.theme_mode:
        st.session_state.theme_mode = mode
        st.rerun()

    # Kode saham dengan auto-suggest dari CSV
    st.subheader("Kode Saham")
    ticker = st.selectbox(
        "Cari atau pilih kode saham",
        options=ID_TICKERS,
        index=ID_TICKERS.index("BBCA.JK") if "BBCA.JK" in ID_TICKERS else 0,
        help="Ketik beberapa huruf (misal: BB, TL, GO) untuk mencari saham.",
    )

    st.info(
        "Gunakan kode Yahoo Finance, misalnya: BBRI.JK, TLKM.JK, GOTO.JK, dll.",
        icon="💡",
    )

# =====================================================
# 8. MAIN CONTENT
# =====================================================
# Header besar
st.markdown(
    f"""
    <div class="header-card">
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="font-size:40px;">📈</div>
            <div>
                <h1 class="app-title">Prediksi Tren Harga Saham Menggunakan LSTM/GRU</h1>
                <p class="app-subtitle">
                    Analisis prediksi tren untuk saham <b>{ticker}</b> berdasarkan data historis dan indikator teknikal.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model, scaler, params
model, scaler, params = load_resources()

if model is None:
    st.error(
        "File model (.keras) atau scaler (.pkl) tidak ditemukan. "
        "Pastikan file `model_lstm_stock_trend.keras`, `scaler_stock_trend.pkl`, "
        "dan `model_params.pkl` sudah di-upload."
    )
else:
    if st.button("🚀 Mulai Analisis", use_container_width=True):
        with st.spinner("Mengambil data pasar & melakukan prediksi..."):
            try:
                # 1. Ambil data dari Yahoo Finance
                df_raw = yf.download(ticker, period="1y", progress=False)

                if len(df_raw) <= 60:
                    st.warning(
                        "Data historis saham tidak cukup untuk melakukan analisis (butuh lebih dari 60 data)."
                    )
                else:
                    # Jika MultiIndex, ratakan
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)

                    # 2. Tambah indikator teknikal
                    df_processed = add_technical_indicators(df_raw)

                    # 3. Siapkan input untuk model
                    features = params.get("feature_columns", [])
                    look_back = params.get("look_back", 30)

                    available_cols = [c for c in features if c in df_processed.columns]
                    if len(available_cols) != len(features):
                        st.warning(
                            "Beberapa fitur yang diharapkan model tidak tersedia. "
                            "Akurasi prediksi mungkin berkurang."
                        )

                    input_data = df_processed[available_cols].values[-look_back:]
                    input_scaled = scaler.transform(input_data)
                    X_input = input_scaled.reshape(1, look_back, len(available_cols))

                    # 4. Prediksi
                    proba = model.predict(X_input)[0]
                    pred_class = int(np.argmax(proba))

                    labels = ["DOWNTREND 📉", "SIDEWAYS ➡️", "UPTREND 🚀"]
                    result_text = labels[pred_class]
                    confidence = float(proba[pred_class] * 100)

                    # ============================
                    #   TAMPILAN METRICS
                    # ============================
                    st.subheader("📊 Hasil Prediksi")

                    col1, col2, col3 = st.columns(3)

                    metric_titles = [
                        "Harga Penutupan Terakhir",
                        "Prediksi Tren",
                        "Keyakinan Model",
                    ]
                    metric_values = [
                        f"Rp {df_raw['Close'].iloc[-1]:,.0f}",
                        result_text,
                        f"{confidence:.2f}%",
                    ]

                    for col, title, value in zip(
                        (col1, col2, col3), metric_titles, metric_values
                    ):
                        col.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="font-size:14px;color:#9CA3AF;">{title}</div>
                                <div style="font-size:24px;font-weight:bold;margin-top:4px;">
                                    {value}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # ============================
                    #   GRAFIK HARGA & BOLLINGER
                    # ============================
                    st.subheader("📈 Grafik Pergerakan Harga & Bollinger Bands")

                    plot_data = df_processed.iloc[-100:]

                    fig = go.Figure()

                    # Harga penutupan
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data["Close"],
                            mode="lines",
                            name="Close Price",
                            line=dict(color="#2962FF", width=2),
                        )
                    )

                    # Bollinger Bands
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data["BB_Upper"],
                            mode="lines",
                            name="BB Upper",
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data["BB_Lower"],
                            mode="lines",
                            name="Bollinger Bands",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(41, 98, 255, 0.1)",
                        )
                    )

                    fig.update_layout(
                        template=chart_template,
                        height=480,
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (IDR)",
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=30, b=0),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # ============================
                    #   GRAFIK PROBABILITAS
                    # ============================
                    st.subheader("🧠 Detail Probabilitas Prediksi")

                    probs_df = pd.DataFrame(
                        {
                            "Kategori": ["Downtrend", "Sideways", "Uptrend"],
                            "Probabilitas": proba,
                        }
                    )

                    colors = ["#FF4B4B", "#808495", "#09AB3B"]

                    fig_bar = go.Figure(
                        data=[
                            go.Bar(
                                x=probs_df["Kategori"],
                                y=probs_df["Probabilitas"],
                                marker_color=colors,
                                text=(probs_df["Probabilitas"] * 100).map(
                                    "{:.1f}%".format
                                ),
                                textposition="auto",
                            )
                        ]
                    )

                    fig_bar.update_layout(
                        template=chart_template,
                        height=300,
                        yaxis=dict(range=[0, 1], showgrid=False),
                        margin=dict(l=0, r=0, t=30, b=0),
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")

# =====================================================
# 9. FOOTER
# =====================================================
st.markdown(
    """
<div class="footer">
    © <b>Bagus Darmawan</b> - NPM: <b>202210715059</b> | Universitas Bhayangkara Jakarta Raya
</div>
""",
    unsafe_allow_html=True,
)
