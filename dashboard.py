import streamlit as st
import polars as pl
import plotly.express as px
import io

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Интеллектуальная система анализа продаж")

# =========================
# Загрузка CSV
# =========================
def read_csv_with_encodings(file_bytes, columns):
    encodings = ["utf8", "cp1251", "latin1"]

    for enc in encodings:
        try:
            df = pl.read_csv(
                io.BytesIO(file_bytes),
                encoding=enc,
                columns=columns,
                ignore_errors=True
            )
            return df
        except Exception:
            continue

    raise ValueError("Не удалось прочитать файл")


# =========================
# Обработка
# =========================
@st.cache_data(show_spinner=True)
def process_file(file_bytes, date_col, time_col, total_col, freq):

    df = read_csv_with_encodings(
        file_bytes,
        [date_col, time_col, total_col]
    )

    # datetime
    df = df.with_columns(
        (
            pl.col(date_col).cast(pl.Utf8)
            + " "
            + pl.col(time_col).cast(pl.Utf8)
        ).alias("datetime_str")
    )

    df = df.with_columns(
        pl.col("datetime_str").str.to_datetime(strict=False).alias("datetime")
    )

    df = df.drop_nulls(["datetime"])

    # total numeric
    df = df.with_columns(
        pl.col(total_col).cast(pl.Float64, strict=False)
    )

    # частота
    every_map = {
        "D": "1d",
        "W": "1w",
        "M": "1mo"
    }

    every = every_map[freq]

    result = (
        df.group_by_dynamic(
            "datetime",
            every=every
        )
        .agg(
            pl.col(total_col).sum().alias("sales")
        )
        .sort("datetime")
    )

    return result


# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Загрузите CSV", type="csv")

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    # читаем только заголовки
    preview = pl.read_csv(
        io.BytesIO(file_bytes),
        n_rows=5,
        ignore_errors=True
    )

    st.subheader("Первые строки")
    st.dataframe(preview.to_pandas())

    columns = preview.columns

    date_col = st.selectbox("Колонка даты", columns)
    time_col = st.selectbox("Колонка времени", columns)
    total_col = st.selectbox("Колонка суммы", columns)

    freq = st.selectbox(
        "Агрегация",
        ["D", "W", "M"],
        format_func=lambda x: {
            "D": "День",
            "W": "Неделя",
            "M": "Месяц"
        }[x]
    )

    if st.button("🚀 Построить"):

        result = process_file(
            file_bytes,
            date_col,
            time_col,
            total_col,
            freq
        )

        st.success("Обработка завершена")

        st.subheader("График продаж")

        fig = px.line(
            result.to_pandas(),
            x="datetime",
            y="sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Метрики")

        sales = result["sales"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Среднее", f"{sales.mean():.2f}")
        col2.metric("Максимум", f"{sales.max():.2f}")
        col3.metric("Минимум", f"{sales.min():.2f}")
