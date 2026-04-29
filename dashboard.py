import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import io

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Интеллектуальная система прогнозирования продаж")

# =========================
# ЧТЕНИЕ CSV
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
# ПОДГОТОВКА ДАННЫХ (УЛУЧШЕННАЯ)
# =========================
@st.cache_data(show_spinner=True)
def prepare_ts(file_bytes, date_col, time_col, total_col, freq):

    df = safe_read(file_bytes, [date_col, time_col, total_col])

    pdf = df.to_pandas()

    # -------------------------
    # 1. Очистка строк
    # -------------------------
    pdf[date_col] = pdf[date_col].astype(str).str.strip()
    pdf[time_col] = pdf[time_col].astype(str).str.strip()

    # -------------------------
    # 2. datetime
    # -------------------------
    pdf["datetime"] = pd.to_datetime(
        pdf[date_col] + " " + pdf[time_col],
        errors="coerce",
        infer_datetime_format=True
    )

    pdf = pdf.dropna(subset=["datetime"])

    # -------------------------
    # 3. Числа
    # -------------------------
    pdf[total_col] = (
        pdf[total_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    pdf[total_col] = pd.to_numeric(pdf[total_col], errors="coerce")

    # -------------------------
    # 4. Удаление мусора
    # -------------------------
    pdf = pdf.dropna(subset=[total_col])
    pdf = pdf[pdf[total_col] >= 0]

    # -------------------------
    # 5. Удаление дубликатов
    # -------------------------
    pdf = pdf.drop_duplicates(subset=["datetime"])

    # -------------------------
    # 6. Агрегация
    # -------------------------
    ts = (
        pdf.set_index("datetime")[total_col]
        .resample(freq)
        .sum()
    )

    # -------------------------
    # 7. Заполнение пропусков
    # -------------------------
    ts = ts.fillna(0)

    # -------------------------
    # 8. Удаление выбросов (IQR)
    # -------------------------
    q1 = ts.quantile(0.25)
    q3 = ts.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    ts = ts.clip(lower=lower, upper=upper)

    # -------------------------
    # 9. Логарифмирование
    # -------------------------
    ts_log = np.log1p(ts)

    return ts, ts_log


# =========================
# ПРОГНОЗ
# =========================
def forecast_sarima(ts_log, steps, seasonal_period):

    model = sm.tsa.SARIMAX(
        ts_log,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period)
    )

    res = model.fit(disp=False)

    forecast = res.get_forecast(steps=steps)

    pred_log = forecast.predicted_mean
    conf_log = forecast.conf_int()

    # обратно из логарифма
    pred = np.expm1(pred_log)
    conf = np.expm1(conf_log)

    return pred, conf


# =========================
# ОЦЕНКА
# =========================
def evaluate_model(ts_log, seasonal_period):

    split = int(len(ts_log) * 0.8)

    train = ts_log[:split]
    test = ts_log[split:]

    model = sm.tsa.SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period)
    )

    res = model.fit(disp=False)

    pred_log = res.forecast(steps=len(test))

    # обратно
    pred = np.expm1(pred_log)
    test_real = np.expm1(test)

    # MAPE
    mape = np.mean(np.abs((test_real - pred) / test_real.replace(0, np.nan))) * 100

    # RMSE
    rmse = np.sqrt(np.mean((test_real - pred) ** 2))

    return mape, rmse, pred, test_real


# =========================
# UI
# =========================
file = st.file_uploader("📂 Загрузите CSV", type="csv")

if file:

    file_bytes = file.getvalue()

    preview = safe_read(file_bytes)

    st.subheader("📌 Предпросмотр")
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

    if st.button("🚀 Построить"):

        ts, ts_log = prepare_ts(file_bytes, date_col, time_col, total_col, freq)

        # сезонность
        seasonal_map = {"D": 7, "W": 52, "M": 12}
        seasonal_period = seasonal_map[freq]

        # оценка
        mape, rmse, pred_test, test = evaluate_model(ts_log, seasonal_period)

        st.subheader("📊 Качество модели")

        c1, c2 = st.columns(2)
        c1.metric("MAPE (%)", f"{mape:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")

        # график теста
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test.index, y=test, name="Факт"))
        fig_test.add_trace(go.Scatter(x=pred_test.index, y=pred_test, name="Прогноз"))
        st.plotly_chart(fig_test, use_container_width=True)

        # финальный прогноз
        pred, conf = forecast_sarima(ts_log, horizon, seasonal_period)

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

        # метрики
        st.subheader("📊 Общие показатели")

        c1, c2, c3 = st.columns(3)
        c1.metric("Среднее", f"{ts.mean():.2f}")
        c2.metric("Максимум", f"{ts.max():.2f}")
        c3.metric("Минимум", f"{ts.min():.2f}")
