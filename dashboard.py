import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import os, warnings, gc, time
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fpdf import FPDF

# ---------------------------- Функции метрик ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------------------------- Интерфейс Streamlit ----------------------------
st.set_page_config(layout="wide")
st.title("📈 Прогнозирование продаж (лёгкая версия)")

# Загрузка файла
uploaded = st.file_uploader("Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded is not None:
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Размер файла превышает 150 МБ. Загрузите файл меньшего размера.")
        st.stop()

    # Кодировка
    enc_choice = st.selectbox("Кодировка файла", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'])
    if enc_choice == 'auto':
        raw = uploaded.read()
        try:
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
        except ImportError:
            enc = 'utf-8'
        uploaded.seek(0)
    else:
        enc = enc_choice

    # Быстрая загрузка только нужных столбцов
    try:
        dtype_dict = {
            'date': str, 'time': str, 'category': str, 'product': str,
            'quantity': np.float32, 'price': np.float32, 'total': np.float32   # float32 для экономии памяти
        }
        df = pd.read_csv(
            uploaded,
            encoding=enc,
            usecols=['date','time','category','product','quantity','price','total'],
            dtype=dtype_dict,
            on_bad_lines='skip'
        )
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        st.stop()

    # Проверка столбцов
    required = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in required):
        st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(set(required)-set(df.columns))}")
        st.stop()

    # Проверка на инъекции
    def has_injection(val):
        s = str(val).strip()
        return s.startswith(('=', '+', '-', '@'))
    injection_found = False
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(has_injection).any():
            injection_found = True
            break
    if injection_found:
        st.error("⚠️ Обнаружены ячейки, начинающиеся с '=', '+', '-', '@'. Загрузка остановлена.")
        st.stop()

    # Очистка данных
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    if not time_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    # Числовые колонки уже float32, но на всякий случай
    for c in ['quantity','price','total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("❌ После очистки не осталось данных.")
        st.stop()

    st.success(f"✅ Загружено {len(df)} записей")
    st.subheader("Первые 10 строк")
    st.dataframe(df.head(10))

    # Периодичность
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность агрегации", list(freq_map.keys()))
    freq = freq_map[freq_label]

    # Категория / товар
    categories = ['Все'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Категория", categories)
    if selected_category != 'Все':
        products = ['Все'] + sorted(df[df['category'] == selected_category]['product'].unique().tolist())
    else:
        products = ['Все']
    selected_product = st.selectbox("Товар", products)

    # Горизонт
    horizon = st.slider("Горизонт прогноза (периодов)", 1, 52, 8)

    # Фильтрация
    if selected_category == 'Все':
        df_filtered = df.copy()
    else:
        df_filtered = df[df['category'] == selected_category]
        if selected_product != 'Все':
            df_filtered = df_filtered[df_filtered['product'] == selected_product]
    if df_filtered.empty:
        st.warning("⚠️ Нет данных для выбранной комбинации.")
        st.stop()

    # Кнопка прогноза
    if st.button("🚀 Построить прогноз"):
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # --------------------- Агрегация ---------------------
            status_text.text("Агрегация данных...")
            progress_bar.progress(10)

            # Используем лёгкую агрегацию сразу в resample без создания огромных промежуточных таблиц
            ts = df_filtered.set_index('datetime').resample(freq)['total'].sum()
            # Удаляем отфильтрованный DataFrame, он больше не нужен
            del df_filtered
            gc.collect()

            ts = ts.asfreq(freq)
            ts.interpolate(method='linear', inplace=True)
            ts.bfill(inplace=True)
            ts.ffill(inplace=True)
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"❌ Недостаточно данных (всего {len(ts)} точек). Уменьшите горизонт.")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            progress_bar.progress(30)
            status_text.text("Обучение модели Holt-Winters...")

            # Параметры сезонности
            if freq == 'h':
                sp = 24
            elif freq == 'D':
                sp = 7
            elif freq == 'W-MON':
                sp = 52
            else:
                sp = 12
            if sp >= len(train):
                sp = max(2, len(train)//2)

            # Единственная модель – Holt-Winters
            try:
                model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=sp,
                    initialization_method='estimated'
                ).fit()
                test_pred = model.forecast(horizon)
                rmse_val = np.sqrt(mean_squared_error(test, test_pred))
                mape_val = mape(test, test_pred) * 100
            except Exception as e:
                st.error(f"❌ Ошибка обучения модели: {e}")
                st.stop()

            progress_bar.progress(70)
            status_text.text("Построение прогноза...")

            # Обучение на полном ряде
            full_ts = pd.concat([train, test])
            full_model = ExponentialSmoothing(
                full_ts, trend='add', seasonal='add',
                seasonal_periods=sp,
                initialization_method='estimated'
            ).fit()
            forecast = full_model.forecast(horizon)

            # Определение следующей даты
            if freq == 'MS':
                start = full_ts.index[-1] + pd.DateOffset(months=1)
            elif freq == 'W-MON':
                start = full_ts.index[-1] + pd.DateOffset(weeks=1)
            elif freq == 'h':
                start = full_ts.index[-1] + pd.DateOffset(hours=1)
            elif freq == 'D':
                start = full_ts.index[-1] + pd.DateOffset(days=1)
            else:
                start = full_ts.index[-1] + pd.Timedelta(1, unit=freq)
            future = pd.date_range(start=start, periods=horizon, freq=freq)

            # 90% доверительный интервал
            std_res = np.std(np.array(test) - np.array(test_pred))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            elapsed = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"Готово за {elapsed:.1f} сек.")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # ---------- Вывод результатов ----------
            st.subheader("🏆 Результаты прогнозирования (Holt-Winters)")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse_val:.2f}")
            col2.metric("MAPE", f"{mape_val:.2f}%")

            # Интерактивный график
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Test', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future, y=forecast, name='Forecast', line=dict(color='green')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% CI'))
            split_date = test.index[0]
            fig.add_shape(type='line', x0=split_date, x1=split_date,
                          y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split_date, y=1, yref='paper',
                               text='Прогноз', showarrow=False,
                               xanchor='left', textangle=-90)
            fig.update_layout(title="Прогноз (Holt-Winters)",
                              xaxis_title='Дата', yaxis_title='Сумма (total)',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True,
                            config={'scrollZoom': True, 'displayModeBar': True})

            # Таблица прогноза
            st.subheader("📋 Прогнозные значения")
            forecast_table = pd.DataFrame({
                'Дата': future,
                'Прогноз': forecast,
                'Нижняя граница (90%)': lower,
                'Верхняя граница (90%)': upper
            })
            st.dataframe(forecast_table, use_container_width=True)

            # PDF-отчёт
            if st.button("📄 Скачать PDF-отчёт"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Отчёт о прогнозировании", ln=1, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, f"Модель: Holt-Winters", ln=1)
                pdf.cell(200, 10, f"Категория: {selected_category}, Товар: {selected_product}", ln=1)
                pdf.cell(200, 10, f"Периодичность: {freq_label}", ln=1)
                pdf.cell(200, 10, f"Горизонт: {horizon} периодов", ln=1)
                pdf.cell(200, 10, f"RMSE: {rmse_val:.2f}", ln=1)
                pdf.cell(200, 10, f"MAPE: {mape_val:.2f}%", ln=1)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 9)
                pdf.cell(50, 8, "Дата", 1)
                pdf.cell(40, 8, "Прогноз", 1)
                pdf.cell(40, 8, "Нижняя", 1)
                pdf.cell(40, 8, "Верхняя", 1)
                pdf.ln()
                pdf.set_font("Arial", size=9)
                for i, dt in enumerate(future):
                    pdf.cell(50, 8, dt.strftime("%Y-%m-%d"), 1)
                    pdf.cell(40, 8, f"{forecast[i]:.2f}", 1)
                    pdf.cell(40, 8, f"{lower[i]:.2f}", 1)
                    pdf.cell(40, 8, f"{upper[i]:.2f}", 1)
                    pdf.ln()

                fig_mpl, ax = plt.subplots(figsize=(8,4))
                ax.plot(train.index, train.values, label='Train')
                ax.plot(test.index, test.values, label='Test')
                ax.plot(future, forecast, label='Forecast')
                ax.fill_between(future, lower, upper, alpha=0.2)
                ax.axvline(split_date, color='red', linestyle='--')
                ax.legend()
                buf = BytesIO()
                fig_mpl.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plt.close(fig_mpl)
                pdf.image(buf, x=10, w=190)
                buf.close()

                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="forecast_report.pdf">Скачать PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Ошибка при построении прогноза: {e}")
        finally:
            # Очистка памяти
            del train, test, ts
            gc.collect()
