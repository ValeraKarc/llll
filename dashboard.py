import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# ---------- Вспомогательные функции ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------- Интерфейс ----------
st.set_page_config(page_title="Прогноз продаж", layout="wide")
st.title("📈 Прогнозирование продаж")
st.markdown("Загрузите CSV-файл с данными о продажах и получите прогноз на выбранный период.")

# Боковая панель с подсказками
with st.sidebar:
    st.header("💡 Подсказки")
    st.markdown("- **Обязательные столбцы:** date, time, category, product, quantity, price, total")
    st.markdown("- **Формат даты:** любой (дд.мм.гггг, гггг-мм-дд и т.д.)")
    st.markdown("- **Кодировка:** автоопределение или выберите вручную (обычно utf-8 или cp1251)")
    st.markdown("- **Периодичность:** выберите, с каким интервалом агрегировать продажи")
    st.markdown("- **Горизонт:** на сколько периодов вперёд строить прогноз")
    st.markdown("- **Быстрый режим:** только Holt‑Winters (работает мгновенно)")
    st.markdown("- **Расширенный режим:** добавляется Random Forest (точнее, но дольше)")

uploaded = st.file_uploader("📂 Загрузите CSV-файл (до 150 МБ)", type=["csv"])

if uploaded is not None:
    # Проверка размера
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Размер файла превышает 150 МБ. Пожалуйста, загрузите файл меньшего размера.")
        st.stop()

    # Выбор кодировки
    enc_choice = st.selectbox("📝 Кодировка файла", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'])
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

    # Загрузка данных (только нужные столбцы, экономичные типы)
    try:
        dtypes = {
            'date': str, 'time': str, 'category': str, 'product': str,
            'quantity': np.float32, 'price': np.float32, 'total': np.float32
        }
        df = pd.read_csv(
            uploaded,
            encoding=enc,
            usecols=['date','time','category','product','quantity','price','total'],
            dtype=dtypes,
            on_bad_lines='skip'
        )
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        st.stop()

    # Проверка наличия всех обязательных столбцов
    required = ['date','time','category','product','quantity','price','total']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(missing)}")
        st.stop()

    # Защита от инъекций
    def has_injection(val):
        s = str(val).strip()
        return s.startswith(('=', '+', '-', '@'))
    injection_detected = False
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(has_injection).any():
            injection_detected = True
            break
    if injection_detected:
        st.error("⚠️ Обнаружены потенциально опасные конструкции (ячейки начинаются с '=', '+', '-', '@'). Загрузка остановлена.")
        st.stop()

    # Очистка данных
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_is_empty = df['time'].str.replace(r'[\s\.]', '', regex=True).eq('').all()
    if not time_is_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    for col in ['quantity','price','total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("❌ После очистки не осталось данных. Проверьте файл.")
        st.stop()

    st.success(f"✅ Данные загружены: {len(df)} записей")
    with st.expander("🔍 Просмотреть первые 10 строк"):
        st.dataframe(df.head(10))

    # Настройки прогноза
    col1, col2, col3 = st.columns(3)
    with col1:
        freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
        freq_label = st.selectbox("📅 Периодичность", list(freq_map.keys()), index=3)  # по умолчанию месяц
        freq = freq_map[freq_label]
    with col2:
        categories = ['Все'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("🏷️ Категория", categories)
    with col3:
        if selected_category != 'Все':
            products = ['Все'] + sorted(df[df['category'] == selected_category]['product'].unique().tolist())
        else:
            products = ['Все']
        selected_product = st.selectbox("📦 Товар", products)

    horizon = st.slider("🔮 Горизонт прогноза (периодов)", min_value=1, max_value=52, value=5)

    # Режим модели
    use_ml = st.checkbox("🔄 Расширенный режим (добавить Random Forest)", value=False,
                         help="Может улучшить точность на длинных рядах, но требует больше времени.")

    # Фильтрация данных
    if selected_category == 'Все':
        df_filtered = df.copy()
    else:
        df_filtered = df[df['category'] == selected_category]
        if selected_product != 'Все':
            df_filtered = df_filtered[df_filtered['product'] == selected_product]

    if df_filtered.empty:
        st.warning("⚠️ Для выбранной комбинации категории/товара нет данных.")
        st.stop()

    # Основная кнопка
    if st.button("🚀 Построить прогноз"):
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # ----- Агрегация -----
            status_text.text("⏳ Агрегация данных...")
            progress_bar.progress(10)

            # Преобразуем float32 -> float64 для resample (statsmodels требует float64)
            ts = df_filtered.set_index('datetime')['total'].astype(np.float64).resample(freq).sum()
            del df_filtered          # освобождаем память
            gc.collect()

            ts = ts.asfreq(freq)
            ts.interpolate(method='linear', inplace=True)
            ts.bfill(inplace=True)
            ts.ffill(inplace=True)
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"❌ Недостаточно данных для прогноза (всего {len(ts)} точек). Уменьшите горизонт или измените периодичность.")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            # ----- Параметры сезонности -----
            if freq == 'h':
                sp = 24
            elif freq == 'D':
                sp = 7
            elif freq == 'W-MON':
                sp = 52
            else:  # 'MS'
                sp = 12
            if sp >= len(train):
                sp = max(2, len(train)//2)

            results = {}

            # ----- Holt-Winters -----
            status_text.text("🧠 Обучение Holt-Winters...")
            progress_bar.progress(30)
            try:
                hw = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=sp,
                    initialization_method='estimated'
                ).fit()
                pred_hw = hw.forecast(horizon)
                rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))
                mape_hw = mape(test, pred_hw) * 100
                results['Holt-Winters'] = {
                    'rmse': rmse_hw,
                    'mape': mape_hw,
                    'pred_test': pred_hw,
                    'model': hw
                }
            except Exception as e:
                st.warning(f"⚠️ Holt-Winters не удалось обучить: {e}")

            # ----- Random Forest (лёгкий) -----
            if use_ml:
                status_text.text("🌲 Обучение Random Forest...")
                progress_bar.progress(50)

                # Используем всего 3 лага
                lags = min(3, len(train)//2)
                X_train = pd.DataFrame(index=train.index)
                for lag in range(1, lags+1):
                    X_train[f'lag_{lag}'] = train.shift(lag)
                y_train = train.copy()
                # Удаляем строки с NaN, возникшие из-за лагов
                valid_idx = ~X_train.isna().any(axis=1)
                X_train = X_train.loc[valid_idx]
                y_train = y_train.loc[valid_idx]

                if len(X_train) > 5:
                    rf = RandomForestRegressor(
                        n_estimators=30,
                        max_depth=5,
                        random_state=42,
                        n_jobs=-1  # использовать все ядра
                    )
                    rf.fit(X_train, y_train)

                    # Рекурсивный прогноз на тестовый период
                    test_pred = []
                    # Берём последние lags значений из тренировочной части
                    history = y_train.iloc[-lags:].values.tolist()
                    for _ in range(len(test)):
                        feat = {f'lag_{i+1}': history[-i-1] for i in range(lags)}
                        X_row = pd.DataFrame([feat])
                        pred = rf.predict(X_row)[0]
                        test_pred.append(pred)
                        history.append(pred)
                    test_pred = np.array(test_pred)
                    rmse_rf = np.sqrt(mean_squared_error(test, test_pred))
                    mape_rf = mape(test, test_pred) * 100
                    results['Random Forest'] = {
                        'rmse': rmse_rf,
                        'mape': mape_rf,
                        'pred_test': test_pred,
                        'model': rf
                    }
                else:
                    st.warning("⚠️ Недостаточно данных для обучения Random Forest после создания лагов.")

            if not results:
                st.error("❌ Ни одна модель не обучилась. Проверьте данные.")
                st.stop()

            # ----- Выбор лучшей модели -----
            best_name = min(results, key=lambda k: results[k]['rmse'])
            best = results[best_name]

            status_text.text(f"🏆 Лучшая модель: {best_name}")
            progress_bar.progress(60)

            # ----- Финальное обучение на ВСЕХ данных -----
            full_ts = pd.concat([train, test])

            if best_name == 'Holt-Winters':
                full_model = ExponentialSmoothing(
                    full_ts,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=sp,
                    initialization_method='estimated'
                ).fit()
                forecast = full_model.forecast(horizon)
            else:  # Random Forest
                lags = min(3, len(full_ts)//2)
                X_full = pd.DataFrame(index=full_ts.index)
                for lag in range(1, lags+1):
                    X_full[f'lag_{lag}'] = full_ts.shift(lag)
                y_full = full_ts.copy()
                valid_idx = ~X_full.isna().any(axis=1)
                X_full = X_full.loc[valid_idx]
                y_full = y_full.loc[valid_idx]

                full_model = RandomForestRegressor(
                    n_estimators=30,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
                full_model.fit(X_full, y_full)

                # Рекурсивный прогноз на будущее
                history = y_full.iloc[-lags:].values.tolist()
                forecast = []
                for _ in range(horizon):
                    feat = {f'lag_{i+1}': history[-i-1] for i in range(lags)}
                    X_row = pd.DataFrame([feat])
                    pred = full_model.predict(X_row)[0]
                    forecast.append(pred)
                    history.append(pred)
                forecast = np.array(forecast)

            # ----- Доверительный интервал (90%) -----
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            # ----- Построение будущих дат -----
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
            progress_bar.progress(90)
            status_text.text(f"✅ Готово за {elapsed:.1f} сек.")
            time.sleep(0.5)
            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()

            # ----- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ -----
            st.subheader(f"🏆 Лучшая модель: **{best_name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{best['rmse']:,.2f}")
            with col2:
                st.metric("MAPE", f"{best['mape']:.2f}%")

            # Интерактивный график
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values,
                                     name='Исторические данные', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values,
                                     name='Тестовые данные', line=dict(color='#ff7f0e')))
            fig.add_trace(go.Scatter(x=future, y=forecast,
                                     name='Прогноз', line=dict(color='#2ca02c', dash='solid')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(44,160,44,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% доверительный интервал'
            ))
            # Разделяющая линия
            split_date = test.index[0]
            fig.add_vline(x=split_date, line_width=1, line_dash="dash", line_color="red",
                          annotation_text="Начало прогноза", annotation_position="top left")
            fig.update_layout(
                title=f"Прогноз продаж ({best_name})",
                xaxis_title="Дата",
                yaxis_title="Сумма продаж",
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # Таблица прогнозных значений
            st.subheader("📊 Таблица прогнозных значений")
            forecast_table = pd.DataFrame({
                'Дата': future.strftime('%Y-%m-%d'),
                'Прогноз': forecast.round(2),
                'Нижняя граница (90%)': lower.round(2),
                'Верхняя граница (90%)': upper.round(2)
            })
            st.dataframe(forecast_table, use_container_width=True)

            # PDF-отчёт
            if st.button("📄 Скачать отчёт (PDF)"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Отчёт о прогнозировании", ln=1, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, f"Модель: {best_name}", ln=1)
                pdf.cell(200, 10, f"Категория: {selected_category}, Товар: {selected_product}", ln=1)
                pdf.cell(200, 10, f"Периодичность: {freq_label} | Горизонт: {horizon} периодов", ln=1)
                pdf.cell(200, 10, f"RMSE: {best['rmse']:,.2f} | MAPE: {best['mape']:.2f}%", ln=1)
                pdf.ln(5)
                # Таблица
                pdf.set_font("Arial", 'B', 9)
                pdf.cell(50, 8, "Дата", 1)
                pdf.cell(40, 8, "Прогноз", 1)
                pdf.cell(40, 8, "Нижняя гр.", 1)
                pdf.cell(40, 8, "Верхняя гр.", 1)
                pdf.ln()
                pdf.set_font("Arial", size=9)
                for i, dt in enumerate(future):
                    pdf.cell(50, 8, dt.strftime('%Y-%m-%d'), 1)
                    pdf.cell(40, 8, f"{forecast[i]:,.2f}", 1)
                    pdf.cell(40, 8, f"{lower[i]:,.2f}", 1)
                    pdf.cell(40, 8, f"{upper[i]:,.2f}", 1)
                    pdf.ln()
                # График в PDF
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
            st.error(f"❌ Во время выполнения прогноза произошла ошибка: {e}")
        finally:
            # Окончательная очистка памяти
            del train, test, ts
            gc.collect()
