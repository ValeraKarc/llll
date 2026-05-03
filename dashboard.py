import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка бэкенда Matplotlib для Streamlit Cloud
import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ML и статистические модели
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

from fpdf import FPDF

# ===================== Вспомогательные функции =====================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def create_lag_features(series, lags, freq_str):
    df_feat = pd.DataFrame(index=series.index)
    for lag in range(1, lags + 1):
        df_feat[f'lag_{lag}'] = series.shift(lag)
    df_feat['rolling_mean_3'] = series.rolling(window=3).mean()
    freq_lower = str(freq_str).lower()
    if 'h' in freq_lower:
        df_feat['hour'] = series.index.hour
        df_feat['dayofweek'] = series.index.dayofweek
    elif 'd' in freq_lower:
        df_feat['dayofweek'] = series.index.dayofweek
    elif 'w' in freq_lower:
        df_feat['weekofyear'] = series.index.isocalendar().week.astype(int)
    elif 'm' in freq_lower:
        df_feat['month'] = series.index.month
    df_feat = df_feat.dropna()
    return df_feat, series[df_feat.index]

def recursive_forecast(model, initial_series, forecast_dates, lags, freq_str):
    history = initial_series.copy()
    preds = []
    freq_lower = str(freq_str).lower()
    for dt in forecast_dates:
        last_vals = history.iloc[-lags:]
        features = {}
        for i in range(1, lags + 1):
            features[f'lag_{i}'] = last_vals.iloc[-i] if len(last_vals) >= i else np.nan
        features['rolling_mean_3'] = last_vals.iloc[-3:].mean() if len(last_vals) >= 3 else np.mean(last_vals)
        if 'h' in freq_lower:
            features['hour'] = dt.hour
            features['dayofweek'] = dt.dayofweek
        elif 'd' in freq_lower:
            features['dayofweek'] = dt.dayofweek
        elif 'w' in freq_lower:
            features['weekofyear'] = dt.isocalendar().week
        elif 'm' in freq_lower:
            features['month'] = dt.month
        X = pd.DataFrame([features])
        pred = model.predict(X)[0]
        preds.append(pred)
        history = pd.concat([history, pd.Series({dt: pred})])
    return np.array(preds)

def train_and_evaluate_ml(model, train_series, test_index, lags, freq_str):
    X_train_full, y_train_full = create_lag_features(train_series, lags, freq_str)
    if len(X_train_full) == 0:
        return None, None
    model.fit(X_train_full, y_train_full)
    return recursive_forecast(model, train_series, test_index, lags, freq_str)

# ===================== Интерфейс приложения =====================
st.set_page_config(layout="wide")
st.title("🔮 Прогнозирование временных рядов продаж")

encoding_options = ['auto', 'utf-8', 'cp1251', 'latin1', 'iso-8859-1', 'cp1252']
encoding_choice = st.selectbox("Кодировка CSV-файла", encoding_options, index=0)

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

if uploaded_file is not None:
    # ---------- Кодировка ----------
    if encoding_choice == 'auto':
        content = uploaded_file.read()
        try:
            import chardet
            result = chardet.detect(content)
            enc = result['encoding'] or 'utf-8'
            st.info(f"Автоопределена кодировка: {enc}")
        except ImportError:
            st.warning("chardet не установлен, используется UTF-8.")
            enc = 'utf-8'
        uploaded_file.seek(0)
    else:
        enc = encoding_choice

    # ---------- Чтение CSV ----------
    try:
        df = pd.read_csv(uploaded_file, encoding=enc)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    # ---------- Проверка столбцов ----------
    required = ['date', 'time', 'category', 'product', 'quantity', 'price', 'total']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Отсутствуют столбцы: {', '.join(missing)}")
        st.stop()

    # ---------- Парсинг даты и времени (с явным приведением к строке) ----------
    # Гарантируем, что date и time - строки, даже если в CSV они числовые
    date_series = df['date'].astype(str).str.strip()
    time_series = df['time'].astype(str).str.strip()
    # Если время целиком состоит из пробелов/точек, считаем его пустым
    time_is_empty = time_series.str.replace(r'[\s\.]', '', regex=True).eq('').all()

    if not time_is_empty:
        datetime_str = date_series + ' ' + time_series
        df['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(date_series, errors='coerce')

    success_rate = df['datetime'].notna().mean()
    if success_rate < 0.9:
        st.warning(f"Распарсено только {success_rate:.1%} дат. Укажите форматы вручную.")
        with st.expander("⚙️ Ручная настройка форматов", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                date_fmt = st.text_input("Формат даты", value="%Y-%m-%d")
            if not time_is_empty:
                with cols[1]:
                    time_fmt = st.text_input("Формат времени", value="%H:%M:%S")
            if st.button("Применить форматы"):
                parsed_dates = pd.to_datetime(date_series, format=date_fmt, errors='coerce')
                if not time_is_empty:
                    parsed_times = pd.to_datetime(time_series, format=time_fmt, errors='coerce').dt.time
                    df['datetime'] = pd.to_datetime(
                        parsed_dates.dt.strftime('%Y-%m-%d') + ' ' + pd.Series(parsed_times).astype(str),
                        errors='coerce'
                    )
                else:
                    df['datetime'] = parsed_dates
                st.info(f"Распарсено {df['datetime'].notna().mean():.1%}")
        if df['datetime'].notna().sum() == 0:
            st.error("Не удалось распарсить даты.")
            st.stop()

    # ---------- Очистка ----------
    df.dropna(subset=['datetime'], inplace=True)
    for col in ['quantity', 'price', 'total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['quantity', 'price', 'total'], inplace=True)
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("Нет данных после очистки.")
        st.stop()
    st.success(f"✅ Загружено {len(df)} записей")

    # ---------- Настройки прогноза ----------
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность агрегации", list(freq_map.keys()))
    freq = freq_map[freq_label]
    horizon = st.number_input("Горизонт прогноза", min_value=1, max_value=100, value=10, step=1)

    if st.button("🚀 Создать прогноз"):
        with st.spinner("Обучение моделей... Это может занять некоторое время."):
            ts = df.set_index('datetime').resample(freq)['total'].sum().dropna()
            if len(ts) < horizon + 5:
                st.error(f"Недостаточно данных: минимум {horizon+5} точек, а в ряду {len(ts)}.")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            if freq == 'h':
                seasonal_periods = 24
            elif freq == 'D':
                seasonal_periods = 7
            elif freq == 'W-MON':
                seasonal_periods = 52
            else:
                seasonal_periods = 12
            if seasonal_periods >= len(train):
                seasonal_periods = max(2, len(train) // 2)
            lags = min(5, len(train) // 2)

            results = {}

            # Holt-Winters
            try:
                model = ExponentialSmoothing(train, trend='add', seasonal='add',
                                             seasonal_periods=seasonal_periods,
                                             initialization_method='estimated').fit()
                pred_test = model.forecast(horizon)
                rmse = np.sqrt(mean_squared_error(test, pred_test))
                mape = mean_absolute_percentage_error(test, pred_test) * 100
                results['Holt-Winters'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_test, 'model': model}
            except Exception as e:
                st.warning(f"Holt-Winters не обучена: {e}")

            # ARIMA
            if HAS_PMDARIMA:
                try:
                    model = pm.auto_arima(train, seasonal=True, m=seasonal_periods,
                                          suppress_warnings=True, error_action='ignore',
                                          stepwise=True, trace=False)
                    pred_test = model.predict(n_periods=horizon)
                    rmse = np.sqrt(mean_squared_error(test, pred_test))
                    mape = mean_absolute_percentage_error(test, pred_test) * 100
                    results['ARIMA'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_test, 'model': model}
                except Exception as e:
                    st.warning(f"ARIMA не обучена: {e}")
            else:
                st.info("ARIMA пропущена (pmdarima не установлен)")

            # Random Forest
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                pred_rf = train_and_evaluate_ml(rf, train, test.index, lags, freq)
                if pred_rf is not None:
                    rmse = np.sqrt(mean_squared_error(test, pred_rf))
                    mape = mean_absolute_percentage_error(test, pred_rf) * 100
                    results['Random Forest'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_rf, 'model': rf}
            except Exception as e:
                st.warning(f"Random Forest не обучена: {e}")

            # XGBoost
            if HAS_XGB:
                try:
                    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                    pred_xgb = train_and_evaluate_ml(xgb, train, test.index, lags, freq)
                    if pred_xgb is not None:
                        rmse = np.sqrt(mean_squared_error(test, pred_xgb))
                        mape = mean_absolute_percentage_error(test, pred_xgb) * 100
                        results['XGBoost'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_xgb, 'model': xgb}
                except Exception as e:
                    st.warning(f"XGBoost не обучена: {e}")

            # LightGBM
            if HAS_LGB:
                try:
                    lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                    pred_lgb = train_and_evaluate_ml(lgb, train, test.index, lags, freq)
                    if pred_lgb is not None:
                        rmse = np.sqrt(mean_squared_error(test, pred_lgb))
                        mape = mean_absolute_percentage_error(test, pred_lgb) * 100
                        results['LightGBM'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_lgb, 'model': lgb}
                except Exception as e:
                    st.warning(f"LightGBM не обучена: {e}")

            if not results:
                st.error("Ни одна модель не обучилась.")
                st.stop()

        # ---------- Вывод метрик ----------
        best_name = min(results, key=lambda x: results[x]['rmse'])
        st.subheader("🏆 Результаты прогнозирования")
        col1, col2, col3 = st.columns(3)
        col1.metric("Лучшая модель", best_name)
        col2.metric("RMSE на тесте", f"{results[best_name]['rmse']:.2f}")
        col3.metric("MAPE на тесте", f"{results[best_name]['mape']:.2f}%")

        summary_df = pd.DataFrame([
            {'Модель': name, 'RMSE': f"{res['rmse']:.2f}", 'MAPE': f"{res['mape']:.2f}%"}
            for name, res in results.items()
        ]).sort_values('RMSE')
        st.dataframe(summary_df, use_container_width=True)

        # ---------- Выбор модели для графика ----------
        selected_model = st.selectbox("Модель для графика", list(results.keys()),
                                      index=list(results.keys()).index(best_name))
        selected = results[selected_model]

        # ---------- Прогноз на полных данных ----------
        full_ts = pd.concat([train, test])

        # Безопасное создание будущих дат
        if freq == 'MS':
            start_date = full_ts.index[-1] + pd.DateOffset(months=1)
        elif freq == 'W-MON':
            start_date = full_ts.index[-1] + pd.offsets.Week(weekday=0)
        elif freq == 'h':
            start_date = full_ts.index[-1] + pd.Timedelta(hours=1)
        elif freq == 'D':
            start_date = full_ts.index[-1] + pd.DateOffset(days=1)
        else:
            start_date = full_ts.index[-1] + pd.Timedelta(1, unit=freq)
        future_dates = pd.date_range(start=start_date, periods=horizon, freq=freq)

        if selected_model == 'Holt-Winters':
            model_full = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                              seasonal_periods=seasonal_periods,
                                              initialization_method='estimated').fit()
            forecast = model_full.forecast(horizon)
            try:
                pred_obj = model_full.get_prediction(start=future_dates[0], end=future_dates[-1])
                ci = pred_obj.summary_frame(alpha=0.05)
                lower, upper = ci['pi_lower'].values, ci['pi_upper'].values
            except:
                resid_std = np.std(train - selected['pred_test'])
                lower, upper = forecast - 1.96*resid_std, forecast + 1.96*resid_std
        elif selected_model == 'ARIMA':
            model_full = pm.auto_arima(full_ts, seasonal=True, m=seasonal_periods,
                                       suppress_warnings=True, error_action='ignore',
                                       stepwise=True, trace=False)
            forecast, conf_int = model_full.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
            lower, upper = conf_int[:, 0], conf_int[:, 1]
        else:
            model_full = selected['model']
            forecast = recursive_forecast(model_full, full_ts, future_dates, lags, freq)
            resid_std = np.std(np.array(test) - np.array(selected['pred_test']))
            lower, upper = forecast - 1.96*resid_std, forecast + 1.96*resid_std

        # ---------- Интерактивный график ----------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Test', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Forecast', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=np.concatenate([future_dates, future_dates[::-1]]),
                                 y=np.concatenate([upper, lower[::-1]]),
                                 fill='toself', fillcolor='rgba(0,100,80,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='95% доверит. интервал'))
        # Вертикальная линия – дата как строка ISO
        vline_date = pd.Timestamp(test.index[0]).strftime('%Y-%m-%d %H:%M:%S')
        fig.add_vline(x=vline_date, line_dash="dash", line_color="red",
                      annotation_text="Начало прогноза")
        fig.update_layout(title=f"Прогноз ({selected_model})",
                          xaxis_title="Дата", yaxis_title="Сумма (total)",
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True,
                        config={'scrollZoom': True, 'displayModeBar': True})

        # ---------- PDF ----------
        if st.button("📄 Скачать PDF-отчёт"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Отчёт о прогнозировании", ln=1, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, f"Модель: {selected_model}", ln=1)
            pdf.cell(200, 10, f"Периодичность: {freq_label}", ln=1)
            pdf.cell(200, 10, f"Горизонт: {horizon} периодов", ln=1)
            pdf.cell(200, 10, f"RMSE: {selected['rmse']:.2f}", ln=1)
            pdf.cell(200, 10, f"MAPE: {selected['mape']:.2f}%", ln=1)
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(50, 8, "Дата", 1)
            pdf.cell(40, 8, "Прогноз", 1)
            pdf.cell(40, 8, "Ниж.гр.", 1)
            pdf.cell(40, 8, "Верх.гр.", 1)
            pdf.ln()
            pdf.set_font("Arial", size=9)
            for i, dt in enumerate(future_dates):
                pdf.cell(50, 8, dt.strftime("%Y-%m-%d %H:%M"), 1)
                pdf.cell(40, 8, f"{forecast[i]:.2f}", 1)
                pdf.cell(40, 8, f"{lower[i]:.2f}", 1)
                pdf.cell(40, 8, f"{upper[i]:.2f}", 1)
                pdf.ln()

            fig_mpl, ax = plt.subplots(figsize=(8,4))
            ax.plot(train.index, train.values, label='Train')
            ax.plot(test.index, test.values, label='Test')
            ax.plot(future_dates, forecast, label='Forecast')
            ax.fill_between(future_dates, lower, upper, alpha=0.2)
            ax.axvline(test.index[0], color='red', linestyle='--')
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
