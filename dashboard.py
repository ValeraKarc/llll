import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64, time, gc, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------- Праздники РФ ----------------------------
HOLIDAY_DATES = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),  # Новогодние каникулы
    (2, 23),  # День защитника Отечества
    (3, 8),   # Международный женский день
    (5, 1),   # Праздник Весны и Труда
    (5, 9),   # День Победы
    (6, 12),  # День России
    (11, 4),  # День народного единства
}
def is_holiday(dt):
    return (dt.month, dt.day) in HOLIDAY_DATES

# ---------------------------- Метрика ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------------------------- Очистка выбросов ----------------------------
def remove_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clean = series.copy()
    clean[(clean < lower) | (clean > upper)] = np.nan
    return clean.interpolate().bfill().ffill()

# ---------------------------- Обучение ML ----------------------------
def train_ml_model(model, train_series, test_index, lags, freq, holiday_series=None):
    X = pd.DataFrame(index=train_series.index)
    for lag in range(1, lags+1):
        X[f'lag_{lag}'] = train_series.shift(lag)
    if holiday_series is not None:
        X['holiday'] = holiday_series
    y = train_series.copy()
    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 5:
        return None, None, None
    model.fit(X, y)
    test_pred = []
    hist = y.iloc[-lags:].tolist()
    for i, dt in enumerate(test_index):
        feat = {}
        for j in range(lags):
            feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
        if holiday_series is not None:
            feat['holiday'] = 1 if is_holiday(dt) else 0
        X_row = pd.DataFrame([feat])
        pred = model.predict(X_row)[0]
        test_pred.append(pred)
        hist.append(pred)
    return np.array(test_pred), model, X

# ---------------------------- Обработка одного ряда ----------------------------
def process_target(df_f, target_col, freq, horizon):
    ts = df_f.set_index('datetime')[target_col].astype(np.float64).resample(freq).sum()
    ts = ts.asfreq(freq).interpolate().bfill().ffill().dropna()
    if len(ts) < horizon + 5:
        return None
    ts = remove_outliers(ts)
    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]

    sp = {'W-MON':52,'MS':12}[freq]
    if sp >= len(train): sp = max(2, len(train)//2)
    lags = 24 if freq == 'W-MON' else min(6, len(train)//2)

    holiday_series = None
    if freq in ('D','W-MON'):
        holiday_series = pd.Series(
            [1 if is_holiday(d) else 0 for d in train.index],
            index=train.index, dtype=np.int8
        )

    models = {}
    # Holt-Winters
    try:
        hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                  seasonal_periods=sp,
                                  initialization_method='estimated').fit()
        pred = hw.forecast(horizon)
        models['Holt-Winters'] = {
            'rmse': np.sqrt(mean_squared_error(test, pred)),
            'mape': mape(test, pred)*100,
            'pred_test': pred,
            'model': hw
        }
    except Exception as e:
        st.warning(f"Holt-Winters ({target_col}): {e}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
    pred_rf, rf_model, X_rf = train_ml_model(rf, train, test.index, lags, freq, holiday_series)
    if pred_rf is not None:
        models['Random Forest'] = {
            'rmse': np.sqrt(mean_squared_error(test, pred_rf)),
            'mape': mape(test, pred_rf)*100,
            'pred_test': pred_rf,
            'model': rf_model,
            'X_train': X_rf
        }

    # XGBoost
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.05,
                           random_state=42, verbosity=0, n_jobs=-1)
        pred_xgb, xgb_model, X_xgb = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)
        if pred_xgb is not None:
            models['XGBoost'] = {
                'rmse': np.sqrt(mean_squared_error(test, pred_xgb)),
                'mape': mape(test, pred_xgb)*100,
                'pred_test': pred_xgb,
                'model': xgb_model,
                'X_train': X_xgb
            }

    if not models:
        return None

    best_name = min(models, key=lambda k: models[k]['mape'])
    best = models[best_name]

    # Финальный прогноз (полный ряд)
    full_ts = pd.concat([train, test])
    if freq == 'W-MON':
        start_future = full_ts.index[-1] + pd.Timedelta(weeks=1)
    elif freq == 'MS':
        if full_ts.index[-1].month == 12:
            start_future = pd.Timestamp(year=full_ts.index[-1].year+1, month=1, day=1)
        else:
            start_future = pd.Timestamp(year=full_ts.index[-1].year, month=full_ts.index[-1].month+1, day=1)
    else:
        start_future = full_ts.index[-1] + pd.Timedelta(1, unit=freq)
    future = pd.date_range(start=start_future, periods=horizon, freq=freq)

    if best_name == 'Holt-Winters':
        full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                          seasonal_periods=sp,
                                          initialization_method='estimated').fit()
        forecast = full_model.forecast(horizon)
    else:
        # ML
        X_full = pd.DataFrame(index=full_ts.index)
        for lag in range(1, lags+1):
            X_full[f'lag_{lag}'] = full_ts.shift(lag)
        if holiday_series is not None:
            full_hol = pd.Series([1 if is_holiday(d) else 0 for d in full_ts.index],
                                 index=full_ts.index, dtype=np.int8)
            X_full['holiday'] = full_hol
        y_full = full_ts.copy()
        valid = ~X_full.isna().any(axis=1)
        X_full, y_full = X_full.loc[valid], y_full.loc[valid]
        if best_name == 'Random Forest':
            full_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
        else:
            full_model = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.05,
                                      random_state=42, verbosity=0, n_jobs=-1)
        full_model.fit(X_full, y_full)
        future_hol = None
        if freq in ('D','W-MON'):
            future_hol = [1 if is_holiday(d) else 0 for d in future]
        hist = y_full.iloc[-lags:].tolist()
        forecast = []
        for i in range(horizon):
            feat = {}
            for j in range(lags):
                feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
            if future_hol is not None:
                feat['holiday'] = future_hol[i]
            X_row = pd.DataFrame([feat])
            pred = full_model.predict(X_row)[0]
            forecast.append(pred)
            hist.append(pred)
        forecast = np.array(forecast)

    std_res = np.std(np.array(test) - np.array(best['pred_test']))
    lower = forecast - 1.645*std_res
    upper = forecast + 1.645*std_res

    return {
        'train': train, 'test': test, 'future': future,
        'forecast': forecast, 'lower': lower, 'upper': upper,
        'rmse': best['rmse'], 'mape': best['mape'],
        'best_name': best_name, 'models': models, 'sp': sp,
        'lags': lags, 'freq': freq,
        'X_train_for_best': best.get('X_train', None)
    }

# ---------------------------- Интерфейс ----------------------------
st.set_page_config(page_title="Интеллектуальная модель прогнозирования продаж", layout="wide")
st.title("📈 Интеллектуальная модель прогнозирования продаж")
st.markdown("Загрузите CSV-файл с продажами и получите прогноз с автоматическим выбором лучшей модели.")

with st.sidebar:
    st.info("**Обязательные столбцы:** date, time, category, product, quantity, price, total\n\n"
            "**Формат даты:** любой\n"
            "**Кодировка:** авто или вручную")

uploaded = st.file_uploader("📂 Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded:
    if uploaded.size > 150*1024*1024:
        st.error("❌ Файл > 150 МБ"); st.stop()

    raw = uploaded.read()
    try:
        import chardet
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
    except ImportError:
        enc = 'utf-8'
    uploaded.seek(0)

    try:
        df = pd.read_csv(uploaded, encoding=enc,
                         usecols=['date','time','category','product','quantity','price','total'],
                         dtype={'date':str,'time':str,'category':str,'product':str,
                                'quantity':np.float32,'price':np.float32,'total':np.float32})
    except Exception as e:
        st.error(f"Ошибка чтения: {e}"); st.stop()

    required = ['date','time','category','product','quantity','price','total']
    if missing := [c for c in required if c not in df.columns]:
        st.error(f"❌ Отсутствуют столбцы: {', '.join(missing)}"); st.stop()

    for col in df.select_dtypes(object).columns:
        if df[col].astype(str).str.startswith(('=', '+', '-', '@')).any():
            st.error("⚠️ Обнаружены ячейки с инъекциями"); st.stop()

    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    df['datetime'] = pd.to_datetime(df['date'] + (' ' + df['time'] if not time_empty else ''), errors='coerce')
    df.dropna(subset=['datetime','quantity','price','total'], inplace=True)
    df = df[df['total'] > 0].drop_duplicates()
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("Нет данных после очистки"); st.stop()

    st.success(f"✅ {len(df)} записей загружено")
    with st.expander("🔍 Первые 5 строк"):
        st.dataframe(df.head(5))

    col1, col2, col3 = st.columns(3)
    freq_map = {'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = col1.selectbox("Периодичность", list(freq_map.keys()), index=1)
    freq = freq_map[freq_label]
    cats = ['Все'] + sorted(df['category'].unique())
    selected_cat = col2.selectbox("Категория", cats)
    prods = ['Все'] + (sorted(df[df['category']==selected_cat]['product'].unique()) if selected_cat!='Все' else [])
    selected_prod = col3.selectbox("Товар", prods) if prods else None
    horizon = st.slider("Горизонт (периодов)", 1, 52, 5)
    show_advanced = st.checkbox("📊 Расширенная аналитика (включая корреляции, важность признаков, ACF)")

    # Фильтрация
    df_f = df.copy()
    if selected_cat != 'Все': df_f = df_f[df_f['category']==selected_cat]
    if selected_prod and selected_prod != 'Все': df_f = df_f[df_f['product']==selected_prod]
    if df_f.empty:
        st.warning("Нет данных для выбранной комбинации"); st.stop()

    if st.button("🚀 Построить прогноз"):
        start = time.time()
        progress = st.progress(0)
        status = st.empty()

        try:
            status.text("Агрегация..."); progress.progress(5)
            res_total = process_target(df_f, 'total', freq, horizon)
            if res_total is None:
                st.error("Недостаточно данных для прогноза total"); st.stop()

            status.text("Прогноз количества..."); progress.progress(50)
            res_qty = process_target(df_f, 'quantity', freq, horizon)

            progress.progress(90)
            status.text("Формирование результатов...")

            # ---------- Вывод total ----------
            st.subheader(f"🏆 Результаты прогнозирования (сумма продаж) — модель: {res_total['best_name']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{res_total['rmse']:,.2f}")
            col2.metric("MAPE", f"{res_total['mape']:.2f}%")
            other_total = [m for m in res_total['models'] if m != res_total['best_name']]
            if other_total:
                best_other = min(other_total, key=lambda x: res_total['models'][x]['mape'])
                col3.metric(f"Альтернатива: {best_other}", f"MAPE {res_total['models'][best_other]['mape']:.2f}%")

            # График total
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_total['train'].index, y=res_total['train'].values, name='Обучающие', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=res_total['test'].index, y=res_total['test'].values, name='Тестовые', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=res_total['future'], y=res_total['forecast'], name='Прогноз', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=np.concatenate([res_total['future'], res_total['future'][::-1]]),
                                     y=np.concatenate([res_total['upper'], res_total['lower'][::-1]]),
                                     fill='toself', fillcolor='rgba(44,160,44,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'), name='90% CI'))
            split = res_total['test'].index[0]
            fig.add_shape(type='line', x0=split, x1=split, y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split, y=1, yref='paper', text='Прогноз', showarrow=False,
                               xanchor='left', textangle=-90)
            fig.update_layout(title=f'Прогноз суммы продаж ({res_total["best_name"]})', xaxis_title='Дата', yaxis_title='Сумма')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # Таблица прогнозов
            st.subheader("📋 Прогнозные значения")
            table_df = pd.DataFrame({
                'Дата': res_total['future'].strftime('%d-%m-%Y'),
                'Прогноз суммы': res_total['forecast'].round(2),
                'Нижняя граница (90%)': res_total['lower'].round(2),
                'Верхняя граница (90%)': res_total['upper'].round(2)
            })
            if res_qty is not None:
                table_df['Прогноз количества'] = res_qty['forecast'].round(0).astype(int)
                st.caption("Широкий доверительный интервал для суммы объясняется волатильностью данных: чем сильнее колебания продаж, тем больше неопределённость прогноза.")
            st.dataframe(table_df, use_container_width=True)

            # ---------- Расширенная аналитика ----------
            if show_advanced:
                st.subheader("📊 Расширенная аналитика (сумма продаж)")

                # 1. Сравнение моделей
                comp = pd.DataFrame([
                    {'Модель':n, 'RMSE':d['rmse'], 'MAPE':d['mape']} for n,d in res_total['models'].items()
                ]).sort_values('MAPE')
                st.dataframe(comp, use_container_width=True)
                fig_c = make_subplots(rows=1, cols=2, subplot_titles=('RMSE','MAPE'))
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['RMSE'], name='RMSE'), 1, 1)
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['MAPE'], name='MAPE'), 1, 2)
                fig_c.update_layout(showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)

                # 2. Сезонная декомпозиция
                if len(res_total['train']) >= 2*res_total['sp']+10:
                    try:
                        dec = seasonal_decompose(res_total['train'], model='additive', period=res_total['sp'])
                        fd = make_subplots(rows=4, cols=1, subplot_titles=('Наблюдения','Тренд','Сезонность','Остатки'))
                        fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.observed), 1, 1)
                        fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.trend), 2, 1)
                        fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.seasonal), 3, 1)
                        fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.resid), 4, 1)
                        fd.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fd, use_container_width=True)
                    except: pass

                # 3. Остатки на тесте
                best_res = res_total['models'][res_total['best_name']]
                resid = res_total['test'].values - best_res['pred_test']
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=res_total['test'].index, y=resid, name='Остатки'))
                fig_r.add_hline(y=0, line_dash='dash', line_color='red')
                fig_r.update_layout(title='Остатки модели (факт – прогноз) на тестовом периоде')
                st.plotly_chart(fig_r, use_container_width=True)
                st.caption("Остатки показывают разницу между реальными значениями и прогнозом. Если они случайно разбросаны вокруг нуля — модель хорошая.")

                # 4. Гистограмма остатков
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=resid, nbinsx=20, name='Остатки', histnorm='probability density'))
                from scipy.stats import norm
                if len(resid) > 1:
                    mu, std = np.mean(resid), np.std(resid)
                    x = np.linspace(min(resid), max(resid), 100)
                    pdf = norm.pdf(x, mu, std)
                    fig_hist.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Норм. распр.'))
                fig_hist.update_layout(title='Гистограмма остатков', xaxis_title='Ошибка', yaxis_title='Плотность')
                st.plotly_chart(fig_hist, use_container_width=True)

                # 5. Автокорреляция остатков (ACF)
                if len(resid) > 5:
                    acf_vals, confint = acf(resid, nlags=min(10, len(resid)//2), alpha=0.05)
                    fig_acf = go.Figure()
                    for i, val in enumerate(acf_vals):
                        fig_acf.add_shape(type='line', x0=i-0.5, x1=i+0.5, y0=val, y1=val, line=dict(color='blue'))
                    fig_acf.add_hline(y=1.96/np.sqrt(len(resid)), line_dash='dash', line_color='red')
                    fig_acf.add_hline(y=-1.96/np.sqrt(len(resid)), line_dash='dash', line_color='red')
                    fig_acf.update_layout(title='Автокорреляция остатков (ACF)',
                                          xaxis_title='Лаг', yaxis_title='ACF')
                    st.plotly_chart(fig_acf, use_container_width=True)
                    st.caption("Значимые пики ACF указывают на оставшуюся структуру в ошибках.")

                # 6. Важность признаков (если лучшая модель ML)
                if res_total['best_name'] != 'Holt-Winters' and res_total['X_train_for_best'] is not None:
                    model_obj = best_res['model']
                    X_best = res_total['X_train_for_best']
                    if hasattr(model_obj, 'feature_importances_'):
                        importances = model_obj.feature_importances_
                        feat_names = X_best.columns
                        imp_df = pd.DataFrame({'Признак': feat_names, 'Важность': importances}).sort_values('Важность', ascending=True)
                        fig_imp = go.Figure(go.Bar(x=imp_df['Важность'], y=imp_df['Признак'], orientation='h'))
                        fig_imp.update_layout(title='Важность признаков в лучшей модели', xaxis_title='Важность', yaxis_title='')
                        st.plotly_chart(fig_imp, use_container_width=True)

                # 7. Корреляционная матрица лаговых признаков
                if res_total['X_train_for_best'] is not None:
                    X_corr = res_total['X_train_for_best'].copy()
                    y_corr = res_total['train'].loc[X_corr.index]
                    X_corr['target'] = y_corr
                    corr = X_corr.corr()
                    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                                         colorscale='RdBu_r', zmin=-1, zmax=1))
                    fig_corr.update_layout(title='Корреляционная матрица признаков и целевой переменной')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption("Матрица показывает взаимосвязи между лагами, временными метками и продажами. Высокие значения (ближе к ±1) – сильная связь.")

            st.caption(f"⏱️ Прогноз построен за {time.time()-start:.1f} сек.")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
        finally:
            del df_f
            gc.collect()
