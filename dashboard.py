import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import io

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Прогнозирование продаж")

# =========================
# ЧТЕНИЕ CSV (быстро)
# =========================
def safe_read(file_bytes):
    try:
        return pl.read_csv(
            io.BytesIO(file_bytes),
            ignore_errors=True,
            truncate_ragged_lines=True
        )
    except:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        return pl.from_pandas(df)


# =========================
# ПОДГОТОВКА ДАННЫХ
# =========================
@st.cache_data(show_spinner=False)
def prepare_ts(file_bytes, date_col, time_col, total_col, category_col, selected_category, freq):

    df = safe_read(file_bytes)
    pdf = df.to_pandas()

    # datetime
    if pd.api.types.is_numeric_dtype(pdf[date_col]):
        pdf["datetime"] = pd.to_datetime(pdf[date_col], unit="D", origin="1899-12-30")
    else:
        d = pdf[date_col].astype(str).fillna("").str.strip()
        t = pdf[time_col].astype(str).fillna("").str.strip()
        pdf["datetime"] = pd.to_datetime(d + " " + t, errors="coerce")

    pdf = pdf.dropna(subset=["datetime"])

    # total
    pdf[total_col] = pd.to_numeric(
        pdf[total_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    pdf = pdf.dropna(subset=[total_col])
    pdf = pdf[pdf[total_col] >= 0]

    # фильтр категории
    if category_col != "—" and selected_category != "All":
        pdf = pdf[pdf[category_col] == selected_category]

    # агрегация
    ts = (
        pdf.set_index("datetime")[total_col]
        .resample(freq)
        .sum()
        .fillna(0)
    )

    # выбросы
    q1, q3 = ts.quantile(0.25), ts.quantile(0.75)
    ts = ts.clip(lower=q1 - 1.5*(q3-q1), upper=q3 + 1.5*(q3-q1))

    ts_log = np.log1p(ts)

    return ts, ts_log


# =========================
# КЭШ МОДЕЛИ (ускорение)
# =========================
@st.cache_resource
def train_model(ts_log, seasonal):
    model = sm.tsa.SARIMAX(
        ts_log,
        order=(1,1,1),
        seasonal_order=(1,1,1,seasonal)
    )
    return model.fit(disp=False)


# =========================
# ОЦЕНКА
# =========================
def evaluate(ts_log, seasonal):

    split = int(len(ts_log)*0.8)
    train, test = ts_log[:split], ts_log[split:]

    model = sm.tsa.SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,1,1,seasonal)
    ).fit(disp=False)

    pred_log = model.forecast(len(test))

    pred = np.expm1(pred_log)
    test_real = np.expm1(test)

    mape = np.mean(np.abs((test_real - pred) / test_real.replace(0, np.nan))) * 100
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

    cols = preview.columns

    date_col = st.selectbox("Дата", cols)
    time_col = st.selectbox("Время", cols)
    total_col = st.selectbox("Сумма", cols)

    category_col = st.selectbox("Категория", ["—"] + list(cols))

    selected_category = "All"
    if category_col != "—":
        categories = ["All"] + list(preview[category_col].drop_nulls().unique())
        selected_category = st.selectbox("Выбор категории", categories)

    freq = st.selectbox("Агрегация", ["D", "W", "M"])
    horizon = st.slider("Горизонт прогноза", 7, 60, 14)

    if st.button("🚀 Построить прогноз"):

        progress = st.progress(0, text="Старт...")

        with st.spinner("⏳ Идёт обработка данных..."):

            progress.progress(20, text="Подготовка данных...")
            ts, ts_log = prepare_ts(
                file_bytes,
                date_col,
                time_col,
                total_col,
                category_col,
                selected_category,
                freq
            )

            seasonal_map = {"D": 7, "W": 52, "M": 12}
            seasonal = seasonal_map[freq]

            progress.progress(50, text="Оценка модели...")
            mape, rmse, pred_test, test = evaluate(ts_log, seasonal)

            progress.progress(75, text="Обучение модели...")
            model = train_model(ts_log, seasonal)

            progress.progress(90, text="Прогнозирование...")
            f = model.get_forecast(steps=horizon)

            pred = np.expm1(f.predicted_mean)
            conf = np.expm1(f.conf_int())

            progress.progress(100, text="Готово!")

        # =====================
        # МЕТРИКИ
        # =====================
        st.subheader("📊 Качество")

        c1, c2 = st.columns(2)
        c1.metric("MAPE (%)", f"{mape:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")

        # =====================
        # ГРАФИК ТЕСТА
        # =====================
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test.index, y=test, name="Факт"))
        fig_test.add_trace(go.Scatter(x=pred_test.index, y=pred_test, name="Прогноз"))

        st.plotly_chart(fig_test, use_container_width=True)

        # =====================
        # ПРОГНОЗ
        # =====================
        st.subheader("📈 Прогноз")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts, name="История"))
        fig.add_trace(go.Scatter(x=pred.index, y=pred, name="Прогноз"))

        fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,0], name="Нижняя", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,1], name="Верхняя", line=dict(dash="dot")))

        st.plotly_chart(fig, use_container_width=True)
