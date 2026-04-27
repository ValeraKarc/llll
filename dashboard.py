import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecast Dashboard", layout="wide")

st.title("📊 Система прогнозирования продаж")

# =========================
# 1. ЗАГРУЗКА CSV (КЕШ)
# =========================
@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file, low_memory=False)
    return df


# =========================
# 2. ПРЕДОБРАБОТКА + АГРЕГАЦИЯ
# =========================
@st.cache_data(show_spinner=False)
def make_timeseries(df, freq, category):

    df = df.copy()

    # фильтр по категории
    if category != "All":
        df = df[df["category"] == category]

    # быстрый datetime
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    # оставляем только нужное поле
    ts = (
        df.set_index("datetime")["total"]
        .resample(freq)
        .sum()
        .fillna(0)
    )

    return ts


# =========================
# 3. UI
# =========================
file = st.file_uploader("📂 Загрузите CSV файл", type="csv")

if file:

    # загрузка (1 раз)
    df = load_csv(file)

    st.success("Файл загружен")

    # preview
    st.subheader("📌 Первые 10 строк")
    st.dataframe(df.head(10))

    # =========================
    # НАСТРОЙКИ
    # =========================
    st.sidebar.header("⚙️ Параметры")

    categories = ["All"] + list(df["category"].unique())

    category = st.sidebar.selectbox("Категория", categories)
    freq = st.sidebar.selectbox("Агрегация", ["D", "W", "M"])

    # =========================
    # ВРЕМЕННОЙ РЯД
    # =========================
    ts = make_timeseries(df, freq, category)

    # =========================
    # ВИЗУАЛИЗАЦИЯ
    # =========================
    st.subheader("📈 История продаж")
    st.line_chart(ts)

    # =========================
    # СТАТИСТИКА
    # =========================
    st.subheader("📊 Базовая аналитика")

    col1, col2, col3 = st.columns(3)

    col1.metric("Среднее", f"{ts.mean():.2f}")
    col2.metric("Максимум", f"{ts.max():.2f}")
    col3.metric("Минимум", f"{ts.min():.2f}")
