import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fast Dashboard", layout="wide")

st.title("📊 Быстрая загрузка и анализ продаж")

# =========================
# ЗАГРУЗКА + АГРЕГАЦИЯ (СРАЗУ)
# =========================
@st.cache_data(show_spinner=True)
def load_fast(file, date_col, time_col, total_col, freq):

    result = []

    for chunk in pd.read_csv(
        file,
        chunksize=50000,
        usecols=[date_col, time_col, total_col],
        dtype={total_col: "float32"}
    ):

        # быстрый datetime
        chunk["datetime"] = pd.to_datetime(
            chunk[date_col].astype(str) + " " + chunk[time_col].astype(str),
            errors="coerce"
        )

        chunk = chunk.dropna(subset=["datetime"])

        # агрегируем сразу
        ts = (
            chunk.set_index("datetime")[total_col]
            .resample(freq)
            .sum()
        )

        result.append(ts)

    # объединяем куски
    final = pd.concat(result).groupby(level=0).sum()

    return final.sort_index()


# =========================
# UI
# =========================
file = st.file_uploader("📂 Загрузите CSV", type="csv")

if file:

    # читаем только заголовок (очень быстро)
    df_head = pd.read_csv(file, nrows=5)

    st.subheader("📌 Пример данных")
    st.dataframe(df_head)

    columns = list(df_head.columns)

    # выбор колонок
    date_col = st.selectbox("Дата", columns)
    time_col = st.selectbox("Время", columns)
    total_col = st.selectbox("Сумма (total)", columns)

    freq = st.selectbox("Агрегация", ["D", "W", "M"])

    if st.button("🚀 Запустить обработку"):

        # важно: сбрасываем указатель файла
        file.seek(0)

        ts = load_fast(file, date_col, time_col, total_col, freq)

        st.success("Готово")

        st.subheader("📈 График")
        st.line_chart(ts)

        st.subheader("📊 Метрики")

        col1, col2, col3 = st.columns(3)

        col1.metric("Среднее", f"{ts.mean():.2f}")
        col2.metric("Максимум", f"{ts.max():.2f}")
        col3.metric("Минимум", f"{ts.min():.2f}")
