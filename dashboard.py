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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fpdf import FPDF

# ---------------------------- Функции ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------------------------- Интерфейс ----------------------------
st.set_page_config(layout="wide")
st.title("📈 Прогнозирование продаж (оптимизированный ансамбль)")

uploaded = st.file_uploader("Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded is not None:
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Размер файла превышает 150 МБ.")
        st.stop()

    enc_choice = st.selectbox("Кодировка", ['auto','utf-8','cp1251'])
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

    try:
        dtypes = {'date': str, 'time': str, 'category': str, 'product': str,
                  'quantity': np.float32, 'price': np.float32, 'total': np.float32}
        df = pd.read_csv(uploaded, encoding=enc,
                         usecols=['date','time','category','product','quantity','price','total'],
                         dtype=dtypes, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Ошибка чтения: {e}")
        st.stop()

    required = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in required):
        st.error(f"❌ Отсутствуют столбцы: {', '.join(set(required)-set(df.columns))}")
        st.stop()

    # проверка инъекций
    def has_injection(val):
        s = str(val).strip()
        return s.startswith(('=', '+', '-', '@'))
    if any(df[col].astype(str).apply(has_injection).any() for col in df.columns if df[col].dtype == object):
        st.error("⚠️ Обнаружены опасные конструкции. Загрузка остановлена.")
        st.stop()

    # очистка
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    if not time_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    for c in ['quantity','price','total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("Нет данных после очистки.")
        st.stop()
    st.success(f"✅ Загружено {len(df)} строк")
    st.dataframe(df.head(10))

    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность", list(freq_map.keys()))
    freq = freq_map[freq_label]

    cats = ['Все'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Категория", cats)
    if selected_category != 'Все':
        prods = ['Все'] + sorted(df[df['category'] == selected_category]['product'].unique().tolist())
    else:
        prods = ['Все']
    selected_product = st.selectbox("Товар", prods)

    horizon = st.slider("Горизонт прогноза", 1, 52, 8)

    # фильтрация
    if selected_category == 'Все':
        df_filtered = df.copy()
    else:
        df_filtered = df[df['category'] == selected_category]
        if selected_product != 'Все':
            df_filtered = df_filtered[df_filtered['product'] == selected_product]
    if df_filtered.empty:
        st.warning("Нет данных для выбранной комбинации.")
        st.stop()

    # включение ML
    use_ml = st.checkbox("Расширенный режим (добавить Random Forest)", value=False)

    if st.button("🚀 Построить прогноз"):
        start_time = time.time()
        progress = st.progress(0)
        status = st.empty()
        try:
            # ---------- Агрегация (кэшируем) ----------
            @st.cache_data(show_spinner=False)
            def aggregate(df_json, freq, horizon):
                df_local = pd.read_json(df_json)
                ts = df_local.set_index('datetime').resample(freq)['total'].sum()
                ts = ts.asfreq(freq)
                ts.interpolate(method='linear', inplace=True)
                ts.bfill(inplace=True)
                ts.ffill(inplace=True)
                ts.dropna(inplace=True)
                if len(ts) < horizon + 5:
                    return None, None, None
                train = ts.iloc[:-horizon]
                test = ts.iloc[-horizon:]
                return train, test, ts

            status.text("Агрегация данных...")
            progress.progress(10)
            df_json = df_filtered.to_json()
            del df_filtered
            gc.collect()
            train, test, ts = aggregate(df_json, freq, horizon)
            if train is None:
                st.error("Недостаточно данных после агрегации.")
                st.stop()

            # ---------- Параметры ----------
            if freq == 'h': sp = 24
            elif freq == 'D': sp = 7
            elif freq == 'W-MON': sp = 52
            else: sp = 12
            if sp >= len(train): sp = max(2, len(train)//2)

            results = {}

            # Holt-Winters
            status.text("Holt-Winters...")
            progress.progress(25)
            try:
                hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                          seasonal_periods=sp,
                                          initialization_method='estimated').fit()
                pred_hw = hw.forecast(horizon)
                results['Holt-Winters'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred_hw)),
                    'mape': mape(test, pred_hw)*100,
                    'pred_test': pred_hw, 'model': hw
                }
            except Exception as e:
                st.warning(f"Holt-Winters: {e}")

            # Random Forest (только если use_ml)
            if use_ml:
                status.text("Random Forest (лёгкий)...")
                progress.progress(50)
                lags = min(6, len(train)//2)
                X_train = pd.DataFrame(index=train.index)
                for lag in range(1, lags+1):
                    X_train[f'lag_{lag}'] = train.shift(lag)
                y_train = train.copy()
                X_train.dropna(inplace=True)
                y_train = y_train.loc[X_train.index]
                if len(X_train) > 0:
                    rf = RandomForestRegressor(n_estimators=30, max_depth=4,
                                               random_state=42, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    # рекурсивный прогноз
                    pred_rf = []
                    hist = y_train.iloc[-lags:].tolist()
                    for _ in range(len(test)):
                        feat = {f'lag_{i+1}': hist[-i-1] for i in range(lags)}
                        p = rf.predict(pd.DataFrame([feat]))[0]
                        pred_rf.append(p)
                        hist.append(p)
                    pred_rf = np.array(pred_rf)
                    rmse_rf = np.sqrt(mean_squared_error(test, pred_rf))
                    mape_rf = mape(test, pred_rf)*100
                    results['Random Forest'] = {
                        'rmse': rmse_rf, 'mape': mape_rf,
                        'pred_test': pred_rf, 'model': rf
                    }

            if not results:
                st.error("Модели не обучились.")
                st.stop()

            best_name = min(results, key=lambda k: results[k]['rmse'])
            best = results[best_name]

            status.text("Финальный прогноз...")
            progress.progress(80)

            full_ts = pd.concat([train, test])
            # Обучение лучшей модели на всех данных
            if best_name == 'Holt-Winters':
                full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                                  seasonal_periods=sp,
                                                  initialization_method='estimated').fit()
                forecast = full_model.forecast(horizon)
            else:  # Random Forest
                lags = min(6, len(full_ts)//2)
                X_full = pd.DataFrame(index=full_ts.index)
                for lag in range(1, lags+1):
                    X_full[f'lag_{lag}'] = full_ts.shift(lag)
                y_full = full_ts.copy()
                X_full.dropna(inplace=True)
                y_full = y_full.loc[X_full.index]
                full_model = RandomForestRegressor(n_estimators=30, max_depth=4,
                                                   random_state=42, n_jobs=-1)
                full_model.fit(X_full, y_full)
                # рекурсивный прогноз будущего
                hist = y_full.iloc[-lags:].tolist()
                forecast = []
                for _ in range(horizon):
                    feat = {f'lag_{i+1}': hist[-i-1] for i in range(lags)}
                    p = full_model.predict(pd.DataFrame([feat]))[0]
                    forecast.append(p)
                    hist.append(p)
                forecast = np.array(forecast)

            # Доверительный интервал
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            # Будущие даты
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

            elapsed = time.time() - start_time
            progress.progress(100)
            status.text(f"Готово за {elapsed:.1f} сек.")
            time.sleep(0.5)
            progress.empty()
            status.empty()

            # ---------- Вывод ----------
            st.subheader(f"🏆 Лучшая модель: {best_name}")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{best['rmse']:.2f}")
            col2.metric("MAPE", f"{best['mape']:.2f}%")

            # График
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Test', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future, y=forecast, name='Forecast', line=dict(color='green')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'), name='90% CI'))
            split_date = test.index[0]
            fig.add_shape(type='line', x0=split_date, x1=split_date,
                          y0=0, y1=1, yref='paper', line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split_date, y=1, yref='paper', text='Прогноз',
                               showarrow=False, xanchor='left', textangle=-90)
            fig.update_layout(title=f"Прогноз ({best_name})",
                              xaxis_title='Дата', yaxis_title='Сумма (total)',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

            # Таблица
            st.subheader("📋 Прогнозные значения")
            st.dataframe(pd.DataFrame({
                'Дата': future,
                'Прогноз': forecast,
                'Нижняя граница (90%)': lower,
                'Верхняя граница (90%)': upper
            }), use_container_width=True)

            # PDF (опционально)
            if st.button("📄 Скачать PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200,10,"Отчёт о прогнозировании",ln=1,align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.cell(200,10,f"Модель: {best_name}",ln=1)
                pdf.cell(200,10,f"Категория: {selected_category}, Товар: {selected_product}",ln=1)
                pdf.cell(200,10,f"Периодичность: {freq_label}",ln=1)
                pdf.cell(200,10,f"Горизонт: {horizon} периодов",ln=1)
                pdf.cell(200,10,f"RMSE: {best['rmse']:.2f}",ln=1)
                pdf.cell(200,10,f"MAPE: {best['mape']:.2f}%",ln=1)
                pdf.ln(5)
                pdf.set_font("Arial",'B',9)
                pdf.cell(50,8,"Дата",1)
                pdf.cell(40,8,"Прогноз",1)
                pdf.cell(40,8,"Нижняя",1)
                pdf.cell(40,8,"Верхняя",1)
                pdf.ln()
                pdf.set_font("Arial", size=9)
                for i, dt in enumerate(future):
                    pdf.cell(50,8,dt.strftime("%Y-%m-%d"),1)
                    pdf.cell(40,8,f"{forecast[i]:.2f}",1)
                    pdf.cell(40,8,f"{lower[i]:.2f}",1)
                    pdf.cell(40,8,f"{upper[i]:.2f}",1)
                    pdf.ln()
                # график в PDF
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
                st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="forecast.pdf">Скачать PDF</a>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
        finally:
            del train, test, ts
            gc.collect()
