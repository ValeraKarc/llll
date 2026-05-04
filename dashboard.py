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

# Настройка Matplotlib
import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Модели
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
try:
    import pmdarima as pm
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

from fpdf import FPDF

# ---------- Функции ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def create_lag_features(series, lags, freq_str):
    df_feat = pd.DataFrame(index=series.index)
    for lag in range(1, lags+1):
        df_feat[f'lag_{lag}'] = series.shift(lag)
    df_feat['rolling_mean_3'] = series.rolling(3).mean()
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
    df_feat.dropna(inplace=True)
    return df_feat, series[df_feat.index]

def recursive_forecast(model, history_series, forecast_dates, lags, freq_str):
    hist = history_series.copy()
    preds = []
    freq_lower = str(freq_str).lower()
    for dt in forecast_dates:
        last_vals = hist.iloc[-lags:]
        feat = {}
        for i in range(1, lags+1):
            feat[f'lag_{i}'] = last_vals.iloc[-i] if len(last_vals) >= i else np.nan
        feat['rolling_mean_3'] = last_vals[-3:].mean() if len(last_vals) >= 3 else np.mean(last_vals)
        if 'h' in freq_lower:
            feat['hour'] = dt.hour
            feat['dayofweek'] = dt.dayofweek
        elif 'd' in freq_lower:
            feat['dayofweek'] = dt.dayofweek
        elif 'w' in freq_lower:
            feat['weekofyear'] = dt.isocalendar().week
        elif 'm' in freq_lower:
            feat['month'] = dt.month
        X = pd.DataFrame([feat])
        pred = model.predict(X)[0]
        preds.append(pred)
        hist = pd.concat([hist, pd.Series({dt: pred})])
    return np.array(preds)

def train_ml_model(model, train_series, test_index, lags, freq_str):
    X_train, y_train = create_lag_features(train_series, lags, freq_str)
    if len(X_train) == 0:
        return None, None
    model.fit(X_train, y_train)
    return recursive_forecast(model, train_series, test_index, lags, freq_str)

# ---------- Интерфейс ----------
st.set_page_config(layout="wide")
st.title("🔮 Прогнозирование продаж")

encoding_choice = st.selectbox("Кодировка CSV", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'], index=0)
uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")

if uploaded_file is not None:
    # Кодировка
    if encoding_choice == 'auto':
        raw = uploaded_file.read()
        try:
            import chardet
            res = chardet.detect(raw)
            enc = res['encoding'] or 'utf-8'
        except ImportError:
            enc = 'utf-8'
        uploaded_file.seek(0)
    else:
        enc = encoding_choice

    try:
        df = pd.read_csv(uploaded_file, encoding=enc)
    except Exception as e:
        st.error(f"Ошибка чтения: {e}")
        st.stop()

    # Проверка колонок
    needed = ['date','time','category','product','quantity','price','total']
    miss = [c for c in needed if c not in df.columns]
    if miss:
        st.error(f"❌ Отсутствуют столбцы: {', '.join(miss)}")
        st.stop()

    # Дата/время
    date_s = df['date'].astype(str).str.strip()
    time_s = df['time'].astype(str).str.strip()
    time_empty = time_s.str.replace(r'[\s\.]','',regex=True).eq('').all()

    if not time_empty:
        df['datetime'] = pd.to_datetime(date_s + ' ' + time_s, errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(date_s, errors='coerce')

    if df['datetime'].notna().mean() < 0.9:
        st.warning("Распознано < 90% дат. Попробуйте другую кодировку или вручную укажите формат.")
        with st.expander("Ручной формат"):
            col1, col2 = st.columns(2)
            with col1:
                date_fmt = st.text_input("Формат даты", "%Y-%m-%d")
            if not time_empty:
                with col2:
                    time_fmt = st.text_input("Формат времени", "%H:%M:%S")
            if st.button("Применить"):
                parsed_d = pd.to_datetime(date_s, format=date_fmt, errors='coerce')
                if not time_empty:
                    parsed_t = pd.to_datetime(time_s, format=time_fmt, errors='coerce').dt.time
                    df['datetime'] = pd.to_datetime(
                        parsed_d.dt.strftime('%Y-%m-%d') + ' ' + pd.Series(parsed_t).astype(str),
                        errors='coerce')
                else:
                    df['datetime'] = parsed_d
                st.info(f"Распарсено {df['datetime'].notna().mean():.1%}")
        if df['datetime'].notna().sum() == 0:
            st.error("Не удалось распарсить даты.")
            st.stop()

    # Очистка
    df.dropna(subset=['datetime'], inplace=True)
    for c in ['quantity','price','total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("После очистки нет данных.")
        st.stop()
    st.success(f"✅ Загружено {len(df)} строк")

    # Настройки
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность", list(freq_map.keys()))
    freq = freq_map[freq_label]
    horizon = st.number_input("Горизонт прогноза", 1, 100, 10)

    if st.button("🚀 Создать прогноз"):
        ts = df.set_index('datetime').resample(freq)['total'].sum().dropna()
        if len(ts) < horizon + 5:
            st.error(f"Недостаточно данных. Минимум {horizon+5} точек, у вас {len(ts)}.")
            st.stop()

        train = ts.iloc[:-horizon]
        test = ts.iloc[-horizon:]

        # Сезонность
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
        lags = min(5, len(train)//2)

        results = {}

        # Holt-Winters
        try:
            hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                      seasonal_periods=sp,
                                      initialization_method='estimated').fit()
            p = hw.forecast(horizon)
            results['Holt-Winters'] = {
                'rmse': np.sqrt(mean_squared_error(test, p)),
                'mape': mape(test, p)*100,
                'pred_test': p,
                'model': hw
            }
        except Exception as e:
            st.warning(f"Holt-Winters: {e}")

        # ARIMA
        if HAS_ARIMA:
            try:
                arima = pm.auto_arima(train, seasonal=True, m=sp,
                                      suppress_warnings=True, error_action='ignore',
                                      stepwise=True, trace=False, maxiter=30)
                p = arima.predict(n_periods=horizon)
                results['ARIMA'] = {
                    'rmse': np.sqrt(mean_squared_error(test, p)),
                    'mape': mape(test, p)*100,
                    'pred_test': p,
                    'model': arima
                }
            except Exception as e:
                st.warning(f"ARIMA: {e}")

        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            p = train_ml_model(rf, train, test.index, lags, freq)
            if p is not None:
                results['Random Forest'] = {
                    'rmse': np.sqrt(mean_squared_error(test, p)),
                    'mape': mape(test, p)*100,
                    'pred_test': p,
                    'model': rf
                }
        except Exception as e:
            st.warning(f"Random Forest: {e}")

        # XGBoost
        if HAS_XGB:
            try:
                xgb = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, n_jobs=-1)
                p = train_ml_model(xgb, train, test.index, lags, freq)
                if p is not None:
                    results['XGBoost'] = {
                        'rmse': np.sqrt(mean_squared_error(test, p)),
                        'mape': mape(test, p)*100,
                        'pred_test': p,
                        'model': xgb
                    }
            except Exception as e:
                st.warning(f"XGBoost: {e}")

        # LightGBM
        if HAS_LGB:
            try:
                lgb = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1, n_jobs=-1)
                p = train_ml_model(lgb, train, test.index, lags, freq)
                if p is not None:
                    results['LightGBM'] = {
                        'rmse': np.sqrt(mean_squared_error(test, p)),
                        'mape': mape(test, p)*100,
                        'pred_test': p,
                        'model': lgb
                    }
            except Exception as e:
                st.warning(f"LightGBM: {e}")

        if not results:
            st.error("Ни одна модель не обучилась.")
            st.stop()

        best_name = min(results, key=lambda k: results[k]['rmse'])
        best = results[best_name]
        st.subheader("🏆 Результаты")
        col1, col2, col3 = st.columns(3)
        col1.metric("Лучшая модель", best_name)
        col2.metric("RMSE", f"{best['rmse']:.2f}")
        col3.metric("MAPE", f"{best['mape']:.2f}%")

        # Прогноз на полных данных
        full_ts = pd.concat([train, test])
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

        if best_name == 'Holt-Winters':
            full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                              seasonal_periods=sp,
                                              initialization_method='estimated').fit()
            fcst = full_model.forecast(horizon)
            try:
                pred_obj = full_model.get_prediction(start=future[0], end=future[-1])
                ci = pred_obj.summary_frame(alpha=0.05)
                lower, upper = ci['pi_lower'].values, ci['pi_upper'].values
            except:
                std_res = np.std(train - best['pred_test'])
                lower, upper = fcst - 1.96*std_res, fcst + 1.96*std_res
        elif best_name == 'ARIMA':
            full_model = pm.auto_arima(full_ts, seasonal=True, m=sp,
                                       suppress_warnings=True, error_action='ignore',
                                       stepwise=True, trace=False, maxiter=30)
            fcst, conf = full_model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
            lower, upper = conf[:,0], conf[:,1]
        else:
            full_model = best['model']
            fcst = recursive_forecast(full_model, full_ts, future, lags, freq)
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower, upper = fcst - 1.96*std_res, fcst + 1.96*std_res

        # График (исправленная вертикальная линия)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Test', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=future, y=fcst, name='Forecast', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=np.concatenate([future, future[::-1]]),
                                 y=np.concatenate([upper, lower[::-1]]),
                                 fill='toself', fillcolor='rgba(0,100,80,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='95% CI'))
        split_dt = test.index[0]
        fig.add_shape(type='line', x0=split_dt, x1=split_dt, y0=0, y1=1, yref='paper',
                      line=dict(color='red', dash='dash'))
        fig.add_annotation(x=split_dt, y=1, yref='paper', text='Начало прогноза',
                           showarrow=False, xanchor='left', textangle=-90)
        fig.update_layout(title=f"Прогноз ({best_name})", xaxis_title='Дата', yaxis_title='Total',
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        # PDF
        if st.button("📄 Скачать PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200,10,"Отчёт о прогнозировании",ln=1,align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(200,10,f"Модель: {best_name}",ln=1)
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
                pdf.cell(50,8,dt.strftime("%Y-%m-%d %H:%M"),1)
                pdf.cell(40,8,f"{fcst[i]:.2f}",1)
                pdf.cell(40,8,f"{lower[i]:.2f}",1)
                pdf.cell(40,8,f"{upper[i]:.2f}",1)
                pdf.ln()
            fig_mpl, ax = plt.subplots(figsize=(8,4))
            ax.plot(train.index, train.values, label='Train')
            ax.plot(test.index, test.values, label='Test')
            ax.plot(future, fcst, label='Forecast')
            ax.fill_between(future, lower, upper, alpha=0.2)
            ax.axvline(split_dt, color='red', linestyle='--')
            ax.legend()
            buf = BytesIO()
            fig_mpl.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close(fig_mpl)
            pdf.image(buf, x=10, w=190)
            buf.close()
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="forecast.pdf">Скачать PDF</a>',
                        unsafe_allow_html=True)
