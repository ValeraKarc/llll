import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Анализ продаж (быстрая версия)")

# =========================
# НАДЁЖНОЕ ЧТЕНИЕ CSV
# =========================
def safe_read_csv(file_bytes, columns=None):

    encodings = ["utf8", "cp1251", "latin1"]

    for enc in encodings:
        try:
            df = pl.read_csv(
                io.BytesIO(file_bytes),
                encoding=enc,
                columns=columns,
                ignore_errors=True,
                truncate_ragged_lines=True,
                infer_schema_length=0
            )
            return df
        except Exception:
            continue

    # fallback → pandas
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return pl.from_pandas(df)


# =========================
# ОБРАБОТКА
# =========================
@st.cache_data(show_spinner=True)
def process(file_bytes, date_col, time_col, total_col, freq):

    df = safe_read_csv(file_bytes, [date_col, time_col, total_col])

    # datetime
    df = df.with_columns(
        (
            pl.col(date_col).cast(pl.Utf8)
            + " "
            + pl.col(time_col).cast(pl.Utf8)
        ).alias("dt_str")
    )

    df = df.with_columns(
        pl.col("dt_str").str.to_datetime(strict=False).alias("datetime")
    ).drop_nulls(["datetime"])

    # total → число
    df = df.with_columns(
        pl.col(total_col).cast(pl.Float64, strict=False)
    )

    # частота
    freq_map = {"D": "1d", "W": "1w", "M": "1mo"}

    result = (
        df.group_by_dynamic(
            "datetime",
            every=freq_map[freq]
        )
        .agg(pl.col(total_col).sum().alias("sales"))
        .sort("datetime")
    )

    return result


# =========================
# UI
# =========================
file = st.file_uploader("Загрузите CSV", type="csv")

if file:

    file_bytes = file.getvalue()

    # безопасный preview
    preview = safe_read_csv(file_bytes)

    st.subheader("📌 Пример данных")
    st.dataframe(preview.head(10).to_pandas())

    columns = preview.columns

    # выбор колонок
    date_col = st.selectbox("Дата", columns)
    time_col = st.selectbox("Время", columns)
    total_col = st.selectbox("Сумма", columns)

    freq = st.selectbox("Агрегация", ["D", "W", "M"])

    if st.button("🚀 Построить"):

        result = process(
            file_bytes,
            date_col,
            time_col,
            total_col,
            freq
        )

        st.success("Готово")

        df_plot = result.to_pandas()

        fig = px.line(df_plot, x="datetime", y="sales")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Метрики")

        col1, col2, col3 = st.columns(3)

        col1.metric("Среднее", f"{df_plot['sales'].mean():.2f}")
        col2.metric("Максимум", f"{df_plot['sales'].max():.2f}")
        col3.metric("Минимум", f"{df_plot['sales'].min():.2f}")
