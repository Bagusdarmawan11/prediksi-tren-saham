# =====================================================
#   APP STOCK TREND PREDICTION
#   Author    : Bagus Darmawan
#   Deskripsi : Prediksi tren harga saham (LSTM/GRU)
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# -----------------------------------------------------
# 0. KONFIGURASI GLOBAL
# -----------------------------------------------------
MODEL_VERSION = "v1.0 (2025)"

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

# -----------------------------------------------------
# 1. TEMA & CSS
# -----------------------------------------------------
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
        "sidebar": "#111318",  # Sidebar tetap gelap untuk kontras
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

# -----------------------------------------------------
# 2. LOAD TICKERS DARI CSV
# -----------------------------------------------------
@st.cache_data
def load_tickers_from_csv(csv_path: str = "DaftarSaham.csv"):
    """
    Baca file DaftarSaham.csv dan kembalikan:
    - list ticker 'XXXX.JK'
    - mapping ticker -> nama emiten
    """
    try:
        df_codes = pd.read_csv(csv_path)

        # Asumsikan ada kolom 'Code' dan 'Name'
        df_codes["Code"] = df_codes["Code"].astype(str).str.strip().str.upper()
        df_codes["Ticker"] = df_codes["Code"] + ".JK"

        mapping = dict(zip(df_codes["Ticker"], df_codes["Name"]))
        tickers = sorted(mapping.keys())

        # Jika list kosong, pakai default
        if not tickers:
            raise ValueError("Daftar ticker kosong di CSV.")

        return tickers, mapping

    except Exception as e:
        st.sidebar.warning(
            f"Gagal membaca DaftarSaham.csv: {e}. "
            "Menggunakan beberapa ticker default."
        )
        mapping = {
            "BBCA.JK": "Bank Central Asia Tbk",
            "BBRI.JK": "Bank Rakyat Indonesia Tbk",
            "BMRI.JK": "Bank Mandiri Tbk",
            "TLKM.JK": "Telkom Indonesia Tbk",
            "GOTO.JK": "GoTo Gojek Tokopedia Tbk",
        }
        return sorted(mapping.keys()), mapping


ID_TICKERS, NAME_MAP = load_tickers_from_csv()


def format_ticker(t: str) -> str:
    """Format tampil ticker di selectbox: 'BBCA.JK - Bank Central Asia Tbk'."""
    name = NAME_MAP.get(t, "")
    return f"{t} - {name}" if name else t


# -----------------------------------------------------
# 3. LOAD MODEL & SCALER
# -----------------------------------------------------
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


# -----------------------------------------------------
# 4. DATA HARGA DARI YAHOO
# -----------------------------------------------------
@st.cache_data(ttl=300)
def get_price_data(ticker: str):
    """Ambil data harga 1 tahun dari Yahoo Finance."""
    df = yf.download(ticker, period="1y", progress=False)
    return df


# -----------------------------------------------------
# 5. FEATURE ENGINEERING
# -----------------------------------------------------
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


# -----------------------------------------------------
# 6. DISCLAIMER
# -----------------------------------------------------
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

# -----------------------------------------------------
# 7. SIDEBAR (NAVIGASI + PENGATURAN)
# -----------------------------------------------------
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

    # Navigasi halaman
    page = st.radio(
        "📁 Navigasi",
        ["Dashboard Prediksi", "Tentang Model & Cara Pakai"],
        index=0,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pengaturan
    st.header("⚙️ Pengaturan")

    mode = st.radio("Tema Aplikasi", ["Light", "Dark"], horizontal=True)
    if mode != st.session_state.theme_mode:
        st.session_state.theme_mode = mode
        st.rerun()

    st.subheader("Kode Saham")

    ticker = st.selectbox(
        "Cari atau pilih kode saham",
        options=ID_TICKERS,
        index=ID_TICKERS.index("BBCA.JK") if "BBCA.JK" in ID_TICKERS else 0,
        format_func=format_ticker,
        help="Ketik beberapa huruf (misal: BB, TL, GO) untuk mencari saham.",
    )

    st.info(
        "Kode menggunakan format Yahoo Finance, misalnya: BBRI.JK, TLKM.JK, GOTO.JK.",
        icon="💡",
    )

# -----------------------------------------------------
# 8. HALAMAN 1: DASHBOARD PREDIKSI
# -----------------------------------------------------
if page == "Dashboard Prediksi":

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
                    <p style="font-size:11px;color:#9CA3AF;margin-top:6px;">
                        Model {MODEL_VERSION} &middot; Output berupa klasifikasi <i>Downtrend</i>, <i>Sideways</i>, atau <i>Uptrend</i>.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model
    model, scaler, params = load_resources()

    if model is None:
        st.error(
            "File model (.keras) atau scaler (.pkl) tidak ditemukan.\n\n"
            "Pastikan file `model_lstm_stock_trend.keras`, `scaler_stock_trend.pkl`, "
            "dan `model_params.pkl` berada dalam folder yang sama dengan aplikasi."
        )

    else:
        if st.button("🚀 Mulai Analisis", use_container_width=True):
            with st.spinner("Mengambil data pasar & melakukan prediksi..."):
                try:
                    # -------------------------------------------------
                    # 1. Ambil data harga
                    # -------------------------------------------------
                    df_raw = get_price_data(ticker)

                    if df_raw is None or df_raw.empty:
                        st.warning(
                            "Data untuk kode saham ini tidak ditemukan di Yahoo Finance."
                        )
                        st.stop()

                    if "Close" not in df_raw.columns:
                        st.warning("Kolom 'Close' tidak ditemukan pada data harga.")
                        st.stop()

                    if df_raw["Close"].isna().all():
                        st.warning(
                            "Data harga penutupan untuk saham ini belum tersedia / tidak lengkap."
                        )
                        st.stop()

                    if len(df_raw) <= 60:
                        st.warning(
                            "Data historis saham tidak cukup untuk melakukan analisis "
                            "(butuh lebih dari 60 data)."
                        )
                        st.stop()

                    # Flatten MultiIndex kalau ada
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)

                    last_date = df_raw.index[-1].strftime("%d %B %Y")

                    # -------------------------------------------------
                    # 2. Tambah indikator teknikal
                    # -------------------------------------------------
                    df_processed = add_technical_indicators(df_raw)

                    # -------------------------------------------------
                    # 3. Siapkan input untuk model (VERSI AMAN)
                    # -------------------------------------------------
                    features = params.get("feature_columns", None)
                    look_back = int(params.get("look_back", 30))

                    # pastikan features jadi list biasa
                    if features is None:
                        features = []
                    elif isinstance(features, (pd.Index, pd.Series)):
                        features = features.tolist()
                    elif not isinstance(features, (list, tuple)):
                        features = list(features)

                    if len(features) == 0:
                        st.error(
                            "Parameter `feature_columns` tidak ditemukan atau kosong di `model_params.pkl`."
                        )
                        st.stop()

                    available_cols = [c for c in features if c in df_processed.columns]

                    if len(available_cols) == 0:
                        st.error(
                            "Tidak ada satupun kolom fitur yang ditemukan di data setelah penambahan indikator.\n"
                            "Pastikan nama indikator (SMA_10, EMA_10, dll.) sama seperti saat training."
                        )
                        st.stop()

                    if len(available_cols) != len(features):
                        st.warning(
                            "Beberapa fitur yang diharapkan model tidak tersedia pada data. "
                            "Akurasi prediksi mungkin berkurang.\n\n"
                            f"Fitur yang digunakan saat ini: {available_cols}"
                        )

                    if len(df_processed) < look_back:
                        st.warning(
                            "Data setelah penambahan indikator teknikal tidak cukup "
                            f"untuk window look_back = {look_back}."
                        )
                        st.stop()

                    input_data = df_processed[available_cols].values[-look_back:]

                    # Scaling dengan pengecekan error
                    try:
                        input_scaled = scaler.transform(input_data)
                    except Exception as e:
                        st.error(
                            "Terjadi masalah saat menyiapkan fitur untuk model. "
                            "Pastikan scaler dan fitur sama dengan saat training.\n\n"
                            f"Detail error: {e}"
                        )
                        st.stop()

                    X_input = input_scaled.reshape(1, look_back, len(available_cols))

                    # -------------------------------------------------
                    # 4. Prediksi
                    # -------------------------------------------------
                    proba = model.predict(X_input)[0]
                    pred_class = int(np.argmax(proba))

                    labels_with_icon = ["DOWNTREND 📉", "SIDEWAYS ➡️", "UPTREND 🚀"]
                    labels_plain = ["Downtrend", "Sideways", "Uptrend"]

                    result_text = labels_with_icon[pred_class]
                    result_plain = labels_plain[pred_class]
                    confidence = float(proba[pred_class] * 100)
                    prob_uptrend = float(proba[2] * 100)

                    # -------------------------------------------------
                    # 5. METRIC CARDS
                    # -------------------------------------------------
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

                    # Ringkasan interpretasi
                    st.markdown(
                        f"""
                        **Ringkasan singkat**

                        - Data terakhir yang digunakan: **{last_date}**  
                        - Model memperkirakan tren utama saat ini: **{result_plain}**  
                        - Tingkat keyakinan model terhadap prediksi ini: **{confidence:.1f}%**  
                        - Probabilitas khusus untuk **Uptrend**: **{prob_uptrend:.1f}%**  

                        > Catatan: hasil ini bersifat estimasi dan bukan rekomendasi beli / jual.
                        """
                    )

                    st.divider()

                    # -------------------------------------------------
                    # 6. GRAFIK HARGA & BOLLINGER
                    # -------------------------------------------------
                    st.subheader("📈 Grafik Pergerakan Harga & Bollinger Bands")

                    plot_data = df_processed.iloc[-100:]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data["Close"],
                            mode="lines",
                            name="Close Price",
                            line=dict(color="#2962FF", width=2),
                        )
                    )
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

                    # -------------------------------------------------
                    # 7. GRAFIK RSI
                    # -------------------------------------------------
                    st.subheader("📉 Indikator RSI (14)")

                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data["RSI_14"],
                            mode="lines",
                            name="RSI 14",
                        )
                    )

                    fig_rsi.add_hrect(
                        y0=30,
                        y1=70,
                        fillcolor="rgba(148,163,184,0.22)",
                        line_width=0,
                    )

                    fig_rsi.update_layout(
                        template=chart_template,
                        height=260,
                        yaxis=dict(range=[0, 100]),
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # -------------------------------------------------
                    # 8. GRAFIK PROBABILITAS
                    # -------------------------------------------------
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



# -----------------------------------------------------
# 9. HALAMAN 2: TENTANG MODEL & CARA PAKAI
# -----------------------------------------------------
else:
    st.markdown(
        f"""
        <div class="header-card">
            <div style="display:flex;align-items:center;gap:16px;">
                <div style="font-size:40px;">📚</div>
                <div>
                    <h1 class="app-title">Tentang Model & Cara Menggunakan Aplikasi</h1>
                    <p class="app-subtitle">
                        Penjelasan singkat mengenai arsitektur model, data yang digunakan,
                        serta langkah-langkah penggunaan aplikasi.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Arsitektur Model")

    st.markdown(
        f"""
- Model menggunakan arsitektur **Recurrent Neural Network (RNN)** dengan kombinasi:
  - Layer **LSTM** dan/atau **GRU**
  - Beberapa dense layer di bagian akhir untuk klasifikasi.
- Output model berupa **3 kelas tren harga**:
  1. **Downtrend** – kecenderungan harga bergerak turun.
  2. **Sideways** – harga bergerak mendatar dalam rentang tertentu.
  3. **Uptrend** – kecenderungan harga bergerak naik.
- Model dilatih menggunakan data historis harga saham harian beserta berbagai
  **indikator teknikal**:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI 14)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Log return harga penutupan.

- Versi model yang digunakan saat ini: **{MODEL_VERSION}**.
        """
    )

    st.markdown("### 📂 Data yang Digunakan")

    st.markdown(
        """
- Sumber data: **Yahoo Finance** melalui library `yfinance`.
- Periode default data: **1 tahun terakhir** dari tanggal hari ini.
- Harga yang digunakan:
  - Open, High, Low, Close (OHLC)
  - Volume
- Beberapa baris awal akan terbuang karena perhitungan indikator (misalnya RSI, SMA).
- Aplikasi membutuhkan minimal **> 60 data harian** setelah semua indikator dihitung.
        """
    )

    st.markdown("### 🧾 Cara Menggunakan Aplikasi")

    st.markdown(
        """
1. **Pilih Kode Saham**
   - Gunakan dropdown **"Kode Saham"** di sidebar.
   - Ketik beberapa huruf kode (misalnya `BB`, `TL`, `GO`) untuk menampilkan daftar saham yang mendekati.
   - Format kode mengikuti standard **Yahoo Finance**: `XXXX.JK`.

2. **Atur Tema Tampilan (Opsional)**
   - Pilih **Light** atau **Dark** di bagian "Tema Aplikasi" di sidebar.

3. **Mulai Analisis**
   - Pastikan halaman **"Dashboard Prediksi"** sedang aktif.
   - Klik tombol **"🚀 Mulai Analisis"** di halaman utama.
   - Aplikasi akan:
     - Mengambil data harga 1 tahun terakhir dari Yahoo Finance.
     - Menghitung indikator teknikal.
     - Menyiapkan data sesuai format input model.
     - Menjalankan prediksi tren.

4. **Membaca Hasil Prediksi**
   - Bagian **"Hasil Prediksi"** menampilkan:
     - Harga penutupan terakhir.
     - Kategori tren (Downtrend / Sideways / Uptrend).
     - Tingkat kepercayaan (confidence) model.
   - Di bawahnya terdapat:
     - Grafik harga + Bollinger Bands.
     - Grafik **RSI (14)** untuk melihat kondisi jenuh beli/jual.
     - Grafik batang probabilitas untuk tiap kategori tren.

5. **Mengganti Saham**
   - Ubah pilihan ticker di sidebar, kemudian klik kembali **"Mulai Analisis"**.

> Penting: Hasil prediksi **bukan rekomendasi investasi**. Selalu kombinasikan
> dengan analisis fundamental, berita terkini, dan manajemen risiko pribadi.
        """
    )

    st.markdown("### ⚠️ Batasan & Saran Pengembangan")

    st.markdown(
        """
- Model tidak mempertimbangkan:
  - Berita perusahaan, sentimen pasar, maupun faktor makroekonomi.
  - Corporate action (stock split, right issue, dll.) secara eksplisit.
- Beberapa ide pengembangan lanjutan:
  - Menambahkan fitur **backtest** strategi.
  - Menambah horizon waktu (prediksi 5 hari / 10 hari ke depan).
  - Menggabungkan data sentimen dari berita atau media sosial.
  - Menyimpan log prediksi untuk dianalisis kembali.
        """
    )

# -----------------------------------------------------
# 10. FOOTER
# -----------------------------------------------------
st.markdown(
    """
<div class="footer">
    © <b>Bagus Darmawan</b> - NPM: <b>202210715059</b> | Universitas Bhayangkara Jakarta Raya
</div>
""",
    unsafe_allow_html=True,
)
