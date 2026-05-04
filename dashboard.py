import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly.graph_objects as go
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

# ------------------- Вспомогательные функции -------------------
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

def train_and_evaluate_ml(model, train_series, test_index, lags, freq_str):
    X_train, y_train = create_lag_features(train_series, lags, freq_str)
    if len(X_train) == 0:
        return None, None
    model.fit(X_train, y_train)
    preds = recursive_forecast(model, train_series, test_index, lags, freq_str)
    return preds

# ------------------- Интерфейс приложения -------------------
st.set_page_config(layout="wide")
st.title("Прогнозирование (ансамбль моделей) с очисткой данных")

enc_choice = st.selectbox("Кодировка", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'], index=0)
uploaded = st.file_uploader("CSV-файл", type="csv")

if uploaded is not None:
    # Кодировка
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
        df = pd.read_csv(uploaded, encoding=enc)
    except Exception as e:
        st.error(f"Ошибка чтения: {e}")
        st.stop()

    # Проверка обязательных колонок
    required = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in required):
        st.error(f"Нет обязательных столбцов: {', '.join(required)}")
        st.stop()

    # Обработка даты и времени
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

    # Очистка данных
    df.drop_duplicates(inplace=True)

    Q1 = df['total'].quantile(0.25)
    Q3 = df['total'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['total'] >= lower_bound) & (df['total'] <= upper_bound)]

    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("Нет данных после очистки")
        st.stop()

    st.success(f"Данных после очистки: {len(df)} строк")

    # Настройки прогноза
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность", list(freq_map.keys()))
    freq = freq_map[freq_label]
    horizon = st.slider("Горизонт прогноза", 1, 52, 8)

    if st.button("🚀 Создать прогноз"):
        with st.spinner("Обработка и обучение моделей..."):
            # Агрегация
            ts = df.set_index('datetime').resample(freq)['total'].sum()
            ts = ts.asfreq(freq)
            ts.interpolate(method='linear', inplace=True)
            ts.bfill(inplace=True)
            ts.ffill(inplace=True)
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"Недостаточно данных, осталось {len(ts)} точек")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            # Параметры лагов
            lags = min(12, len(train)//2)

            results = {}

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            pred_rf = train_and_evaluate_ml(rf, train, test.index, lags, freq)
            if pred_rf is not None:
                rmse_rf = np.sqrt(mean_squared_error(test, pred_rf))
                mape_rf = mape(test, pred_rf) * 100
                results['Random Forest'] = {'rmse': rmse_rf, 'mape': mape_rf, 'pred_test': pred_rf, 'model': rf}

            # XGBoost
            if HAS_XGB:
                xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
                pred_xgb = train_and_evaluate_ml(xgb, train, test.index, lags, freq)
                if pred_xgb is not None:
                    rmse_xgb = np.sqrt(mean_squared_error(test, pred_xgb))
                    mape_xgb = mape(test, pred_xgb) * 100
                    results['XGBoost'] = {'rmse': rmse_xgb, 'mape': mape_xgb, 'pred_test': pred_xgb, 'model': xgb}

            # LightGBM
            if HAS_LGB:
                lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
                pred_lgbm = train_and_evaluate_ml(lgbm, train, test.index, lags, freq)
                if pred_lgbm is not None:
                    rmse_lgbm = np.sqrt(mean_squared_error(test, pred_lgbm))
                    mape_lgbm = mape(test, pred_lgbm) * 100
                    results['LightGBM'] = {'rmse': rmse_lgbm, 'mape': mape_lgbm, 'pred_test': pred_lgbm, 'model': lgbm}

            if not results:
                st.error("Ни одна модель не обучилась")
                st.stop()

            # Выбор лучшей модели
            best_name = min(results, key=lambda k: results[k]['rmse'])
            best = results[best_name]

            st.subheader(f"🏆 Лучшая модель: {best_name}")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{best['rmse']:.2f}")
            col2.metric("MAPE", f"{best['mape']:.1f}%")

            # Прогноз на полном ряде
            full = pd.concat([train, test])
            best_model = best['model']
            # Обучаем лучшую модель на всех данных
            X_full, y_full = create_lag_features(full, lags, freq)
            if best_name in ['Random Forest']:
                model_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif best_name == 'XGBoost':
                model_full = XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
            else:  # LightGBM
                model_full = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
            model_full.fit(X_full, y_full)

            # Будущие даты
            if freq == 'MS':
                start = full.index[-1] + pd.DateOffset(months=1)
            elif freq == 'W-MON':
                start = full.index[-1] + pd.DateOffset(weeks=1)
            elif freq == 'h':
                start = full.index[-1] + pd.DateOffset(hours=1)
            elif freq == 'D':
                start = full.index[-1] + pd.DateOffset(days=1)
            else:
                start = full.index[-1] + pd.Timedelta(1, unit=freq)
            future = pd.date_range(start=start, periods=horizon, freq=freq)

            forecast = recursive_forecast(model_full, full, future, lags, freq)

            # Доверительный интервал
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.96 * std_res
            upper = forecast + 1.96 * std_res

            # График
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values,
                                     name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values,
                                     name='Test', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future, y=forecast,
                                     name='Forecast', line=dict(color='green')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'))

            split_date = test.index[0]
            fig.add_shape(type='line', x0=split_date, x1=split_date,
                          y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split_date, y=1, yref='paper',
                               text='Прогноз', showarrow=False,
                               xanchor='left', textangle=-90)

            fig.update_layout(title=f'Прогноз ({best_name})',
                              xaxis_title='Дата', yaxis_title='Сумма (total)',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True,
                            config={'scrollZoom': True, 'displayModeBar': True})

            # PDF-отчёт
            if st.button("📄 Скачать PDF"):
                from fpdf import FPDF
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
                    pdf.cell(40,8,f"{forecast[i]:.2f}",1)
                    pdf.cell(40,8,f"{lower[i]:.2f}",1)
                    pdf.cell(40,8,f"{upper[i]:.2f}",1)
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
                href = f'<a href="data:application/pdf;base64,{b64}" download="forecast.pdf">Скачать PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
