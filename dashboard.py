import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
from io import BytesIO
import base64
import matplotlib, matplotlib.pyplot as plt, time, gc
matplotlib.use('Agg')
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fpdf import FPDF

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

st.set_page_config(page_title="Интеллектуальная модель прогнозирования продаж", layout="wide")
st.title("📈 Интеллектуальная модель прогнозирования продаж")

uploaded = st.file_uploader("Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded:
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Файл > 150 МБ"); st.stop()

    enc = st.selectbox("Кодировка", ['auto', 'utf-8', 'cp1251'])
    if enc == 'auto':
        raw = uploaded.read()
        try:
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
        except: enc = 'utf-8'
        uploaded.seek(0)

    try:
        df = pd.read_csv(uploaded, encoding=enc,
                         usecols=['date', 'time', 'category', 'product', 'quantity', 'price', 'total'],
                         dtype={'date': str, 'time': str, 'category': str, 'product': str,
                                'quantity': np.float32, 'price': np.float32, 'total': np.float32})
    except Exception as e:
        st.error(f"Ошибка чтения: {e}"); st.stop()

    if missing := [c for c in ['date', 'time', 'category', 'product', 'quantity', 'price', 'total'] if c not in df.columns]:
        st.error(f"❌ Нет столбцов: {', '.join(missing)}"); st.stop()

    for col in df.select_dtypes(object).columns:
        if df[col].astype(str).str.startswith(('=', '+', '-', '@')).any():
            st.error("⚠️ Опасные символы в начале ячеек"); st.stop()

    df['datetime'] = pd.to_datetime(df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(),
                                    errors='coerce')
    df.dropna(subset=['datetime', 'quantity', 'price', 'total'], inplace=True)
    df = df[df['total'] > 0].drop_duplicates()
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("❌ Нет данных после очистки"); st.stop()

    st.success(f"✅ {len(df)} строк")
    col1, col2, col3 = st.columns(3)
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq = freq_map[col1.selectbox("Периодичность", list(freq_map.keys()), index=3)]
    cats = ['Все'] + sorted(df['category'].unique())
    selected_cat = col2.selectbox("Категория", cats)
    prods = ['Все'] + (sorted(df[df['category'] == selected_cat]['product'].unique()) if selected_cat != 'Все' else [])
    selected_prod = col3.selectbox("Товар", prods) if prods else None
    horizon = st.slider("Горизонт (периодов)", 1, 52, 5)

    df_f = df.copy()
    if selected_cat != 'Все': df_f = df_f[df_f['category'] == selected_cat]
    if selected_prod and selected_prod != 'Все': df_f = df_f[df_f['product'] == selected_prod]
    if df_f.empty:
        st.warning("Нет данных для выбранной комбинации"); st.stop()

    if st.button("🚀 Построить прогноз"):
        st_progress = st.progress(0)
        st_status = st.empty()
        try:
            st_status.text("Агрегация..."); st_progress.progress(10)
            ts = df_f.set_index('datetime')['total'].astype(np.float64).resample(freq).sum()
            del df_f; gc.collect()
            ts = ts.asfreq(freq).interpolate().bfill().ffill().dropna()
            if len(ts) < horizon + 5:
                st.error(f"Мало данных (есть {len(ts)} точек)"); st.stop()

            train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]
            sp = {'h': 24, 'D': 7, 'W-MON': 52, 'MS': 12}[freq]
            if sp >= len(train): sp = max(2, len(train) // 2)

            st_status.text("Обучение Holt‑Winters..."); st_progress.progress(30)
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp,
                                         initialization_method='estimated').fit()
            pred_test = model.forecast(horizon)
            rmse_val = np.sqrt(mean_squared_error(test, pred_test))
            mape_val = mape(test, pred_test) * 100

            full_ts = pd.concat([train, test])
            full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add', seasonal_periods=sp,
                                              initialization_method='estimated').fit()
            forecast = full_model.forecast(horizon)

            next_date = pd.date_range(start=full_ts.index[-1], periods=2, freq=freq)[-1]
            future = pd.date_range(start=next_date, periods=horizon, freq=freq)

            std_res = np.std(test.values - pred_test.values)
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            st_progress.progress(90); st_status.text("Готово"); time.sleep(0.3)
            st_progress.empty(); st_status.empty()

            st.subheader("Результаты прогнозирования")
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{rmse_val:,.2f}")
            c2.metric("MAPE", f"{mape_val:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Обучающие данные'))
            fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Тестовые данные'))
            fig.add_trace(go.Scatter(x=future, y=forecast, name='Прогноз'))
            fig.add_trace(go.Scatter(x=np.concatenate([future, future[::-1]]),
                                     y=np.concatenate([upper, lower[::-1]]),
                                     fill='toself', fillcolor='rgba(44,160,44,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'), name='90% дов. интервал'))
            fig.add_shape(type='line', x0=test.index[0], x1=test.index[0], y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=test.index[0], y=1, yref='paper', text='Начало прогноза',
                               showarrow=False, xanchor='left', textangle=-90)
            fig.update_layout(title='Прогноз продаж', xaxis_title='Дата', yaxis_title='Сумма продаж')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            st.subheader("Таблица прогнозных значений")
            st.dataframe(pd.DataFrame({
                'Дата': future.strftime('%Y-%m-%d'),
                'Прогноз': forecast.round(2),
                'Нижняя граница': lower.round(2),
                'Верхняя граница': upper.round(2)
            }), use_container_width=True)

            if st.button("📄 Скачать отчёт (PDF)"):
                pdf = FPDF()
                pdf.add_page()
                pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                pdf.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
                pdf.set_font('DejaVu', '', 14)
                pdf.cell(0, 10, 'Отчёт о прогнозировании', ln=1, align='C')
                pdf.ln(10)
                pdf.set_font('DejaVu', '', 12)
                pdf.cell(0, 10, f'Модель: экспоненциальное сглаживание Хольта-Винтерса', ln=1)
                pdf.cell(0, 10, f'Периодичность: {freq_label} | Горизонт: {horizon} периодов', ln=1)
                pdf.cell(0, 10, f'Категория: {selected_cat} | Товар: {selected_prod}', ln=1)
                pdf.cell(0, 10, f'RMSE: {rmse_val:,.2f} | MAPE: {mape_val:.2f}%', ln=1)
                pdf.ln(10)

                pdf.set_font('DejaVu', 'B', 10)
                pdf.cell(50, 8, 'Дата', 1)
                pdf.cell(40, 8, 'Прогноз', 1)
                pdf.cell(40, 8, 'Нижняя граница', 1)
                pdf.cell(40, 8, 'Верхняя граница', 1)
                pdf.ln()
                pdf.set_font('DejaVu', '', 10)
                for i, dt in enumerate(future):
                    pdf.cell(50, 8, dt.strftime('%Y-%m-%d'), 1)
                    pdf.cell(40, 8, f"{forecast[i]:,.2f}", 1)
                    pdf.cell(40, 8, f"{lower[i]:,.2f}", 1)
                    pdf.cell(40, 8, f"{upper[i]:,.2f}", 1)
                    pdf.ln()

                fig_mpl, ax = plt.subplots(figsize=(8, 4))
                ax.plot(train.index, train.values, label='Обучающие')
                ax.plot(test.index, test.values, label='Тестовые')
                ax.plot(future, forecast, label='Прогноз')
                ax.fill_between(future, lower, upper, alpha=0.2)
                ax.axvline(test.index[0], color='red', linestyle='--')
                ax.legend()
                buf = BytesIO()
                fig_mpl.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plt.close(fig_mpl)
                pdf.image(buf, x=10, w=190)
                buf.close()

                pdf_bytes = pdf.output()
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="forecast_report.pdf">Скачать PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
        finally:
            del train, test, ts; gc.collect()
