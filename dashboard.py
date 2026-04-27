import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

st.set_page_config(page_title="Forecast System", layout="wide")

st.title("📊 Прогноз продаж для малого бизнеса")

file = st.file_uploader("Загрузите CSV", type="csv")

# =========================
# PDF ФУНКЦИЯ
# =========================
def create_pdf(metrics_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Отчет по прогнозированию продаж", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(metrics_text, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


if file:

    df = pd.read_csv(file)

    # =========================
    # ФИЛЬТР КАТЕГОРИИ
    # =========================
    category = st.selectbox("Категория", ["All"] + list(df["category"].unique()))

    if category != "All":
        df = df[df["category"] == category]

    # =========================
    # НАСТРОЙКИ
    # =========================
    freq = st.selectbox("Агрегация", ["D", "W", "M"])
    horizon = st.slider("Горизонт прогноза", 7, 60, 14)

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("datetime")

    ts = df.set_index("datetime")["total"].resample(freq).sum()

    st.subheader("📈 История")
    st.line_chart(ts)

    # =========================
    # МОДЕЛИ
    # =========================
    def run_sarima(series):
        model = sm.tsa.SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7))
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=horizon).predicted_mean
        return forecast

    def run_naive(series):
        return pd.Series([series.mean()] * horizon,
                         index=pd.date_range(series.index[-1], periods=horizon+1, freq=freq)[1:])

    # =========================
    # ПРОГНОЗЫ
    # =========================
    sarima_pred = run_sarima(ts)
    naive_pred = run_naive(ts)

    # =========================
    # МЕТРИКИ
    # =========================
    sarima_rmse = np.sqrt(mean_squared_error(ts[-len(sarima_pred):], sarima_pred[:len(ts[-len(sarima_pred):])]))
    naive_rmse = np.sqrt(mean_squared_error(ts[-len(naive_pred):], naive_pred[:len(ts[-len(naive_pred):])]))

    best_model = "SARIMA" if sarima_rmse < naive_rmse else "Naive"

    st.subheader("📊 Сравнение моделей")
    st.write(f"SARIMA RMSE: {sarima_rmse:.2f}")
    st.write(f"Naive RMSE: {naive_rmse:.2f}")
    st.success(f"Лучшая модель: {best_model}")

    # =========================
    # ГРАФИК
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts.index, y=ts, name="История"))
    fig.add_trace(go.Scatter(x=sarima_pred.index, y=sarima_pred, name="SARIMA"))
    fig.add_trace(go.Scatter(x=naive_pred.index, y=naive_pred, name="Naive"))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # PDF ЭКСПОРТ
    # =========================
    metrics_text = f"""
    Лучшая модель: {best_model}<br>
    SARIMA RMSE: {sarima_rmse:.2f}<br>
    Naive RMSE: {naive_rmse:.2f}
    """

    pdf = create_pdf(metrics_text)

    st.download_button(
        label="📄 Скачать PDF отчет",
        data=pdf,
        file_name="forecast_report.pdf",
        mime="application/pdf"
    )
