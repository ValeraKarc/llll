import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from io import BytesIO
import base64
import traceback
import warnings
warnings.filterwarnings('ignore')

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

# Генерация PDF
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================== Вспомогательные функции ==============================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def create_lag_features(series, lags, freq_str):
    """Создаёт признаки для ML-моделей на основе одного временного ряда."""
    df_feat = pd.DataFrame(index=series.index)
    for lag in range(1, lags + 1):
        df_feat[f'lag_{lag}'] = series.shift(lag)
    df_feat['rolling_mean_3'] = series.rolling(window=3).mean()
    
    # Добавляем временные признаки в зависимости от частоты
    if 'h' in freq_str.lower():  # часовые данные
        df_feat['hour'] = series.index.hour
        df_feat['dayofweek'] = series.index.dayofweek
    elif 'd' in freq_str.lower():  # дневные данные
        df_feat['dayofweek'] = series.index.dayofweek
    elif 'w' in freq_str.lower():  # недельные данные
        df_feat['weekofyear'] = series.index.isocalendar().week.astype(int)
    elif 'm' in freq_str.lower():  # месячные данные
        df_feat['month'] = series.index.month
    
    df_feat = df_feat.dropna()
    return df_feat, series[df_feat.index]

def recursive_forecast(model, initial_series, forecast_dates, lags, freq_str):
    """Рекурсивный прогноз ML-модели на заданные даты."""
    history = initial_series.copy()
    preds = []
    for dt in forecast_dates:
        last_vals = history.iloc[-lags:]
        features = {}
        for i in range(1, lags + 1):
            features[f'lag_{i}'] = last_vals.iloc[-i] if len(last_vals) >= i else np.nan
        if len(last_vals) >= 3:
            features['rolling_mean_3'] = last_vals.iloc[-3:].mean()
        else:
            features['rolling_mean_3'] = np.mean(last_vals)
        
        # Добавляем временные признаки
        if 'h' in freq_str.lower():
            features['hour'] = dt.hour
            features['dayofweek'] = dt.dayofweek
        elif 'd' in freq_str.lower():
            features['dayofweek'] = dt.dayofweek
        elif 'w' in freq_str.lower():
            features['weekofyear'] = dt.isocalendar().week
        elif 'm' in freq_str.lower():
            features['month'] = dt.month
            
        X = pd.DataFrame([features])
        pred = model.predict(X)[0]
        preds.append(pred)
        history = pd.concat([history, pd.Series({dt: pred})])
    return np.array(preds)

def train_and_evaluate_ml(model, train_series, test_index, lags, freq_str):
    """Обучение ML модели и рекурсивный прогноз на тестовый период."""
    X_train_full, y_train_full = create_lag_features(train_series, lags, freq_str)
    if len(X_train_full) == 0:
        return None, None
    model.fit(X_train_full, y_train_full)
    preds = recursive_forecast(model, train_series, test_index, lags, freq_str)
    return preds

# ================================== Интерфейс Streamlit ==============================
st.set_page_config(layout="wide")
st.title("🔮 Прогнозирование временных рядов продаж")

# Выбор кодировки перед загрузкой файла
encoding_options = ['auto', 'utf-8', 'cp1251', 'latin1', 'iso-8859-1', 'cp1252']
encoding_choice = st.selectbox("Кодировка CSV-файла", encoding_options, index=0)

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if uploaded_file is not None:
    # Определяем кодировку
    if encoding_choice == 'auto':
        content = uploaded_file.read()
        try:
            import chardet
            result = chardet.detect(content)
            enc = result['encoding']
            if enc is None:
                enc = 'utf-8'
            st.info(f"Автоопределена кодировка: {enc}")
        except ImportError:
            st.warning("Библиотека chardet не установлена, используется UTF-8 с заменами.")
            enc = 'utf-8'
        uploaded_file.seek(0)
    else:
        enc = encoding_choice

    # Чтение CSV
    try:
        df = pd.read_csv(uploaded_file, encoding=enc, encoding_errors='replace' if enc == 'utf-8' else 'strict')
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.stop()

    # Проверка обязательных столбцов
    required = ['date', 'time', 'category', 'product', 'quantity', 'price', 'total']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(missing)}")
        st.stop()

    # Гибкий парсинг даты и времени
    date_series = df['date'].astype(str)
    time_series = df['time'].astype(str)
    time_is_empty = time_series.str.replace(r'[\s\.]', '', regex=True).str.len().sum() == 0

    if not time_is_empty:
        datetime_str = date_series + ' ' + time_series
        df['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(date_series, errors='coerce')

    success_rate = df['datetime'].notna().mean()
    if success_rate < 0.9:
        st.warning(f"Автоматически удалось распарсить только {success_rate:.1%} дат/времени. Укажите форматы вручную.")
        with st.expander("⚙️ Ручная настройка форматов", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                date_fmt = st.text_input("Формат даты (Python strftime)", value="%Y-%m-%d",
                                         help="Например: %d.%m.%Y, %m/%d/%Y, %Y-%m-%d")
            if not time_is_empty:
                with col2:
                    time_fmt = st.text_input("Формат времени", value="%H:%M:%S",
                                             help="Например: %H:%M, %I:%M %p, %H:%M:%S")
            if st.button("Применить форматы"):
                parsed_dates = pd.to_datetime(date_series, format=date_fmt, errors='coerce')
                if not time_is_empty:
                    parsed_times = pd.to_datetime(time_series, format=time_fmt, errors='coerce').dt.time
                    df['datetime'] = pd.to_datetime(
                        parsed_dates.dt.strftime('%Y-%m-%d') + ' ' + parsed_times.astype(str),
                        errors='coerce'
                    )
                else:
                    df['datetime'] = parsed_dates
                success_rate = df['datetime'].notna().mean()
                st.info(f"Новый процент распарсенных: {success_rate:.1%}")
        if df['datetime'].notna().sum() == 0:
            st.error("Не удалось распарсить даты даже после ручной настройки. Проверьте данные.")
            st.stop()

    # Очистка данных
    try:
        df.dropna(subset=['datetime'], inplace=True)
        for col in ['quantity', 'price', 'total']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['quantity', 'price', 'total'], inplace=True)
        df.sort_values('datetime', inplace=True)
        if df.empty:
            st.error("После очистки не осталось данных.")
            st.stop()
        st.success(f"✅ Загружено {len(df)} записей после очистки.")
    except Exception as e:
        st.error(f"Ошибка обработки данных: {e}")
        st.stop()

        # ===================== Блок создания прогноза =====================
    # Настройки прогноза (теперь внутри блока, чтобы состояние сохранялось)
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность агрегации", list(freq_map.keys()))
    freq = freq_map[freq_label]
    horizon = st.number_input("Горизонт прогноза (количество периодов)", min_value=1, max_value=100, value=10, step=1)

    # Кнопка запуска всего прогнозирования
    if st.button("🚀 Создать прогноз"):
        # Агрегация временного ряда
        ts = df.set_index('datetime').resample(freq)['total'].sum().dropna()
        if len(ts) < horizon + 5:
            st.error(f"Недостаточно данных. Длина ряда {len(ts)} точек, а нужно минимум {horizon + 5}.")
            st.stop()

        train = ts.iloc[:-horizon]
        test = ts.iloc[-horizon:]
        st.write(f"Тренировочный период: {train.index.min()} – {train.index.max()} ({len(train)} отсчетов)")
        st.write(f"Тестовый период: {test.index.min()} – {test.index.max()} ({len(test)} отсчетов)")

        # Параметры сезонности и лагов
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

        # 1. Holt-Winters
        try:
            model_hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                            seasonal_periods=seasonal_periods, initialization_method='estimated')
            fitted_hw = model_hw.fit()
            pred_hw = fitted_hw.forecast(horizon)
            rmse = np.sqrt(mean_squared_error(test, pred_hw))
            mape = mean_absolute_percentage_error(test, pred_hw) * 100
            results['Holt-Winters'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_hw, 'model': fitted_hw}
        except Exception as e:
            st.warning(f"Holt-Winters не обучена: {e}")

        # 2. ARIMA
        if HAS_PMDARIMA:
            try:
                arima_model = pm.auto_arima(train, seasonal=True, m=seasonal_periods,
                                            suppress_warnings=True, error_action='ignore',
                                            stepwise=True, trace=False)
                pred_arima = arima_model.predict(n_periods=horizon)
                rmse = np.sqrt(mean_squared_error(test, pred_arima))
                mape = mean_absolute_percentage_error(test, pred_arima) * 100
                results['ARIMA'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_arima, 'model': arima_model}
            except Exception as e:
                st.warning(f"ARIMA не обучена: {e}")
        else:
            st.info("pmdarima не установлен, ARIMA пропущена.")

        # 3. Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            pred_rf = train_and_evaluate_ml(rf, train, test.index, lags, freq)
            if pred_rf is not None:
                rmse = np.sqrt(mean_squared_error(test, pred_rf))
                mape = mean_absolute_percentage_error(test, pred_rf) * 100
                results['Random Forest'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_rf, 'model': rf}
        except Exception as e:
            st.warning(f"Random Forest ошибка: {e}")

        # 4. XGBoost
        if HAS_XGB:
            try:
                xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                pred_xgb = train_and_evaluate_ml(xgb, train, test.index, lags, freq)
                if pred_xgb is not None:
                    rmse = np.sqrt(mean_squared_error(test, pred_xgb))
                    mape = mean_absolute_percentage_error(test, pred_xgb) * 100
                    results['XGBoost'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_xgb, 'model': xgb}
            except Exception as e:
                st.warning(f"XGBoost ошибка: {e}")

        # 5. LightGBM
        if HAS_LGB:
            try:
                lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                pred_lgb = train_and_evaluate_ml(lgb, train, test.index, lags, freq)
                if pred_lgb is not None:
                    rmse = np.sqrt(mean_squared_error(test, pred_lgb))
                    mape = mean_absolute_percentage_error(test, pred_lgb) * 100
                    results['LightGBM'] = {'rmse': rmse, 'mape': mape, 'pred_test': pred_lgb, 'model': lgb}
            except Exception as e:
                st.warning(f"LightGBM ошибка: {e}")

        if not results:
            st.error("Ни одна модель не обучилась. Проверьте данные и установленные библиотеки.")
            st.stop()

        best_name = min(results, key=lambda x: results[x]['rmse'])
        best = results[best_name]
        st.subheader(f"🏆 Лучшая модель: {best_name}")
        st.write(f"RMSE на тесте: {best['rmse']:.2f}")
        st.write(f"MAPE на тесте: {best['mape']:.2f}%")

        # Финальный прогноз на полной выборке
        full_ts = pd.concat([train, test])
        # Определяем единицу для приращения при построении future_dates
        if freq == 'h':
            time_unit = 'h'
        elif freq == 'D':
            time_unit = 'D'
        elif freq == 'W-MON':
            time_unit = 'W'
        else:
            time_unit = 'MS'
        future_dates = pd.date_range(start=full_ts.index[-1] + pd.Timedelta(1, unit=time_unit if time_unit != 'MS' else 'D'),
                                     periods=horizon, freq=freq)

        if best_name == 'Holt-Winters':
            model_full = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                              seasonal_periods=seasonal_periods,
                                              initialization_method='estimated').fit()
            forecast = model_full.forecast(horizon)
            try:
                pred_result = model_full.get_prediction(start=future_dates[0], end=future_dates[-1])
                summary = pred_result.summary_frame(alpha=0.05)
                pi_lower = summary['pi_lower'].values
                pi_upper = summary['pi_upper'].values
            except:
                resid_std = np.std(train - results['Holt-Winters']['pred_test'])
                pi_lower = forecast - 1.96 * resid_std
                pi_upper = forecast + 1.96 * resid_std
        elif best_name == 'ARIMA':
            model_full = pm.auto_arima(full_ts, seasonal=True, m=seasonal_periods,
                                       suppress_warnings=True, error_action='ignore',
                                       stepwise=True, trace=False)
            forecast, conf_int = model_full.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
            pi_lower = conf_int[:, 0]
            pi_upper = conf_int[:, 1]
        else:  # ML модели
            model_full = best['model']
            forecast = recursive_forecast(model_full, full_ts, future_dates, lags, freq)
            test_pred = best['pred_test']
            resid_std = np.std(np.array(test) - np.array(test_pred))
            pi_lower = forecast - 1.96 * resid_std
            pi_upper = forecast + 1.96 * resid_std

        # Интерактивный график с поддержкой масштабирования
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Тренировочные данные',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Тестовые данные',
                                 line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Прогноз',
                                 line=dict(color='green')))
        fig.add_trace(go.Scatter(x=np.concatenate([future_dates, future_dates[::-1]]),
                                 y=np.concatenate([pi_upper, pi_lower[::-1]]),
                                 fill='toself', fillcolor='rgba(0,100,80,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='95% доверительный интервал'))
        split_date = test.index[0]
        fig.add_vline(x=split_date, line_dash="dash", line_color="red", annotation_text="Начало прогноза")
        fig.update_layout(title=f"Прогноз по модели {best_name}",
                          xaxis_title="Дата", yaxis_title="Total")
        # Включаем колесо мыши для масштабирования и другие элементы управления
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        # Кнопка генерации PDF отчёта (появляется после прогноза)
        if st.button("📄 Скачать PDF-отчёт"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Отчёт по прогнозированию", ln=1, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Модель: {best_name}", ln=1)
            pdf.cell(200, 10, txt=f"Периодичность: {freq_label}", ln=1)
            pdf.cell(200, 10, txt=f"Горизонт прогноза: {horizon} периодов", ln=1)
            pdf.cell(200, 10, txt=f"RMSE на тесте: {best['rmse']:.2f}", ln=1)
            pdf.cell(200, 10, txt=f"MAPE на тесте: {best['mape']:.2f}%", ln=1)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Таблица прогнозных значений:", ln=1)
            pdf.ln(2)

            pdf.set_font("Arial", 'B', 9)
            pdf.cell(50, 8, "Дата", 1)
            pdf.cell(40, 8, "Прогноз", 1)
            pdf.cell(40, 8, "Нижняя граница", 1)
            pdf.cell(40, 8, "Верхняя граница", 1)
            pdf.ln()
            pdf.set_font("Arial", size=9)
            for i, dt in enumerate(future_dates):
                pdf.cell(50, 8, dt.strftime("%Y-%m-%d %H:%M"), 1)
                pdf.cell(40, 8, f"{forecast[i]:.2f}", 1)
                pdf.cell(40, 8, f"{pi_lower[i]:.2f}", 1)
                pdf.cell(40, 8, f"{pi_upper[i]:.2f}", 1)
                pdf.ln()

            # График для PDF
            fig_mpl, ax = plt.subplots(figsize=(8, 4))
            ax.plot(train.index, train.values, label='Train', color='blue')
            ax.plot(test.index, test.values, label='Test', color='orange')
            ax.plot(future_dates, forecast, label='Forecast', color='green')
            ax.fill_between(future_dates, pi_lower, pi_upper, alpha=0.2, color='green')
            ax.axvline(split_date, color='red', linestyle='--', label='Forecast start')
            ax.legend()
            ax.set_title(f'Forecast: {best_name}')
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
