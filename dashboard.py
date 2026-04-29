import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import io
import numpy as np

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Интеллектуальная система прогнозирования продаж")

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

    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return pl.from_pandas(df)


# =========================
# ПОДГОТОВКА TS
# =========================
@st.cache_data(show_spinner=True)
def prepare_ts(file_bytes, date_col, time_col, total_col, freq):

    df = safe_read(file_bytes, [date_col, time_col, total_col])

    pdf = df.to_pandas()

    # datetime
    pdf["datetime"] = pd.to_datetime(
        pdf[date_col].astype(str) + " " + pdf[time_col].astype(str),
        errors="coerce"
    )

    pdf = pdf.dropna(subset=["datetime"])

    # числовые значения
    pdf[total_col] = pd.to_numeric(pdf[total_col], errors="coerce").fillna(0)

    ts = (
        pdf.set_index("datetime")[total_col]
        .resample(freq)
        .sum()
    )

    ts = ts.dropna()

    return ts


# =========================
# ПРОГНОЗ
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
# ОЦЕНКА МОДЕЛИ
# =========================
def evaluate_model(ts):

    split = int(len(ts) * 0.8)

    train = ts[:split]
    test = ts[split:]

    model = sm.tsa.SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    res = model.fit(disp=False)

    pred = res.forecast(steps=len(test))

    # MAPE
    mape = np.mean(np.abs((test - pred) / test.replace(0, np.nan))) * 100

    # RMSE
    rmse = np.sqrt(np.mean((test - pred) ** 2))

    return mape, rmse, pred, test


# =========================
# UI
# =========================
file = st.file_uploader("📂 Загрузите CSV файл", type="csv")

if file:

    file_bytes = file.getvalue()

    preview = safe_read(file_bytes)

    st.subheader("📌 Предварительный просмотр")
    st.dataframe(preview.head(10).to_pandas())

    columns = preview.columns

    date_col = st.selectbox("Дата", columns)
    time_col = st.selectbox("Время", columns)
    total_col = st.selectbox("Сумма", columns)

    freq = st.selectbox(
        "Агрегация",
        ["D", "W", "M"],
        format_func=lambda x: {
            "D": "День",
            "W": "Неделя",
            "M": "Месяц"
        }[x]
    )

    horizon = st.slider("Горизонт прогноза", 7, 60, 14)

    if st.button("🚀 Построить прогноз"):

        # =========================
        # TS
        # =========================
        ts = prepare_ts(file_bytes, date_col, time_col, total_col, freq)

        # =========================
        # ОЦЕНКА
        # =========================
        mape, rmse, pred_test, test = evaluate_model(ts)

        st.subheader("📊 Качество модели")

        col1, col2 = st.columns(2)

        col1.metric("MAPE (%)", f"{mape:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")

        # график теста
        fig_test = go.Figure()

        fig_test.add_trace(go.Scatter(x=test.index, y=test, name="Факт"))
        fig_test.add_trace(go.Scatter(x=pred_test.index, y=pred_test, name="Прогноз"))

        st.plotly_chart(fig_test, use_container_width=True)

        # =========================
        # ФИНАЛЬНЫЙ ПРОГНОЗ
        # =========================
        pred, conf = forecast_sarima(ts, horizon)

        st.subheader("📈 Прогноз")

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ts.index, y=ts, name="История"))
        fig.add_trace(go.Scatter(x=pred.index, y=pred, name="Прогноз"))

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

        # =========================
        # МЕТРИКИ ДАННЫХ
        # =========================
        st.subheader("📊 Общие показатели")

        c1, c2, c3 = st.columns(3)

        c1.metric("Среднее", f"{ts.mean():.2f}")
        c2.metric("Максимум", f"{ts.max():.2f}")
        c3.metric("Минимум", f"{ts.min():.2f}")
