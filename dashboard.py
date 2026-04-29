import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import io

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Прогнозирование продаж")

# =========================
# ЧТЕНИЕ CSV (устойчивое)
# =========================
def safe_read(file_bytes, columns=None):
    encodings = ["utf8", "cp1251", "latin1"]

    for enc in encodings:
        try:
            return pl.read_csv(
                io.BytesIO(file_bytes),
                encoding=enc,
                columns=columns,
                ignore_errors=True,
                truncate_ragged_lines=True
            )
        except:
            continue

    df = pd.read_csv(io.BytesIO(file_bytes))
    return pl.from_pandas(df)


# =========================
# ОБРАБОТКА + TS
# =========================
@st.cache_data(show_spinner=True)
def prepare_ts(file_bytes, date_col, time_col, total_col, freq):

    df = safe_read(file_bytes, [date_col, time_col, total_col])

    # 👉 datetime через pandas (самый стабильный способ)
    pdf = df.to_pandas()

    pdf["datetime"] = pd.to_datetime(
        pdf[date_col].astype(str) + " " + pdf[time_col].astype(str),
        errors="coerce"
    )

    pdf = pdf.dropna(subset=["datetime"])

    pdf[total_col] = pd.to_numeric(pdf[total_col], errors="coerce").fillna(0)

    ts = (
        pdf.set_index("datetime")[total_col]
        .resample(freq)
        .sum()
    )

    return ts


# =========================
# ПРОГНОЗ (SARIMA)
# =========================
def forecast_sarima(ts, steps):

    model = sm.tsa.SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    res = model.fit(disp=False)

    forecast = res.get_forecast(steps=steps)

    pred = forecast.predicted_mean
    conf = forecast.conf_int()

    return pred, conf


# =========================
# UI
# =========================
file = st.file_uploader("📂 Загрузите CSV", type="csv")

if file:

    file_bytes = file.getvalue()

    preview = safe_read(file_bytes)

    st.subheader("📌 Данные")
    st.dataframe(preview.head(10).to_pandas())

    columns = preview.columns

    date_col = st.selectbox("Дата", columns)
    time_col = st.selectbox("Время", columns)
    total_col = st.selectbox("Сумма", columns)

    freq = st.selectbox("Агрегация", ["D", "W", "M"])
    horizon = st.slider("Горизонт прогноза", 7, 60, 14)

    if st.button("🚀 Построить прогноз"):

        ts = prepare_ts(file_bytes, date_col, time_col, total_col, freq)

        pred, conf = forecast_sarima(ts, horizon)

        # =========================
        # ГРАФИК
        # =========================
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts,
            name="История"
        ))

        fig.add_trace(go.Scatter(
            x=pred.index,
            y=pred,
            name="Прогноз"
        ))

        fig.add_trace(go.Scatter(
            x=conf.index,
            y=conf.iloc[:, 0],
            name="Нижняя граница",
            line=dict(dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=conf.index,
            y=conf.iloc[:, 1],
            name="Верхняя граница",
            line=dict(dash="dot")
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Метрики")

        col1, col2, col3 = st.columns(3)

        col1.metric("Среднее", f"{ts.mean():.2f}")
        col2.metric("Максимум", f"{ts.max():.2f}")
        col3.metric("Минимум", f"{ts.min():.2f}")
