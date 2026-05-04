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

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------- Праздники РФ ----------------------------
RUSSIAN_HOLIDAYS = {
    '2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05','2020-01-06','2020-01-07','2020-01-08',
    '2020-02-23','2020-03-08','2020-05-01','2020-05-09','2020-06-12','2020-11-04',
    '2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05','2021-01-06','2021-01-07','2021-01-08',
    '2021-02-23','2021-03-08','2021-05-01','2021-05-09','2021-06-12','2021-11-04',
    '2022-01-01','2022-01-02','2022-01-03','2022-01-04','2022-01-05','2022-01-06','2022-01-07','2022-01-08',
    '2022-02-23','2022-03-08','2022-05-01','2022-05-09','2022-06-12','2022-11-04',
    '2023-01-01','2023-01-02','2023-01-03','2023-01-04','2023-01-05','2023-01-06','2023-01-07','2023-01-08',
    '2023-02-23','2023-03-08','2023-05-01','2023-05-09','2023-06-12','2023-11-04',
    '2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-06','2024-01-07','2024-01-08',
    '2024-02-23','2024-03-08','2024-05-01','2024-05-09','2024-06-12','2024-11-04',
    '2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05','2025-01-06','2025-01-07','2025-01-08',
    '2025-02-23','2025-03-08','2025-05-01','2025-05-09','2025-06-12','2025-11-04',
    '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06','2026-01-07','2026-01-08',
    '2026-02-23','2026-03-08','2026-05-01','2026-05-09','2026-06-12','2026-11-04',
}
def is_holiday(dt):
    return dt.strftime('%Y-%m-%d') in RUSSIAN_HOLIDAYS

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
        return None, None
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
    return np.array(test_pred), model

# ---------------------------- Функция прогноза одной модели ----------------------------
def forecast_with_model(model_type, train, test, full_ts, lags, freq, holiday_series, horizon, sp):
    """Обучает модель заданного типа на train, возвращает прогноз на будущее."""
    if model_type == 'Holt-Winters':
        model = ExponentialSmoothing(train, trend='add', seasonal='add',
                                     seasonal_periods=sp,
                                     initialization_method='estimated').fit()
        pred_test = model.forecast(horizon)
        # Переобучение на полном ряду
        full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                          seasonal_periods=sp,
                                          initialization_method='estimated').fit()
        forecast = full_model.forecast(horizon)
        return pred_test, forecast, full_model
    else:
        # ML модели (RF или XGBoost)
        if model_type == 'Random Forest':
            base_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
        else:
            base_model = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.05,
                                      random_state=42, verbosity=0, n_jobs=-1)
        # Обучение на train для получения ошибки и лагов
        X_train = pd.DataFrame(index=train.index)
        for lag in range(1, lags+1):
            X_train[f'lag_{lag}'] = train.shift(lag)
        if holiday_series is not None:
            X_train['holiday'] = holiday_series
        y_train = train.copy()
        valid = ~X_train.isna().any(axis=1)
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]
        base_model.fit(X_train, y_train)
        # Прогноз на тест
        test_pred = []
        hist = y_train.iloc[-lags:].tolist()
        for i, dt in enumerate(test.index):
            feat = {}
            for j in range(lags):
                feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
            if holiday_series is not None:
                feat['holiday'] = 1 if is_holiday(dt) else 0
            X_row = pd.DataFrame([feat])
            pred = base_model.predict(X_row)[0]
            test_pred.append(pred)
            hist.append(pred)
        test_pred = np.array(test_pred)

        # Обучение на полном ряду
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
        full_model = base_model.__class__(**base_model.get_params())
        full_model.fit(X_full, y_full)

        # Рекурсивный прогноз будущего
        # Будущие даты
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
        return test_pred, np.array(forecast), full_model

# ---------------------------- Обработка одного ряда (total или quantity) ----------------------------
def process_target(df_f, target_col, freq, horizon, show_advanced):
    """Полный цикл для одного целевого столбца: агрегация, очистка, модели, прогноз.
    Возвращает словарь с результатами или None при ошибке."""
    ts = df_f.set_index('datetime')[target_col].astype(np.float64).resample(freq).sum()
    ts = ts.asfreq(freq).interpolate().bfill().ffill().dropna()
    if len(ts) < horizon + 5:
        return None
    ts = remove_outliers(ts)
    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]

    sp = {'h':24,'D':7,'W-MON':52,'MS':12}[freq]
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
    pred_rf, rf_model = train_ml_model(rf, train, test.index, lags, freq, holiday_series)
    if pred_rf is not None:
        models['Random Forest'] = {
            'rmse': np.sqrt(mean_squared_error(test, pred_rf)),
            'mape': mape(test, pred_rf)*100,
            'pred_test': pred_rf,
            'model': rf_model
        }

    # XGBoost
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.05,
                           random_state=42, verbosity=0, n_jobs=-1)
        pred_xgb, xgb_model = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)
        if pred_xgb is not None:
            models['XGBoost'] = {
                'rmse': np.sqrt(mean_squared_error(test, pred_xgb)),
                'mape': mape(test, pred_xgb)*100,
                'pred_test': pred_xgb,
                'model': xgb_model
            }

    if not models:
        return None

    best_name = min(models, key=lambda k: models[k]['mape'])
    best = models[best_name]

    # Финальный прогноз
    full_ts = pd.concat([train, test])
    # Определяем start_future
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

    # Прогноз в зависимости от типа модели
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
        'best_name': best_name, 'models': models, 'sp': sp
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

    enc = st.selectbox("Кодировка", ['auto','utf-8','cp1251'])
    if enc == 'auto':
        raw = uploaded.read()
        try:
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
        except: enc = 'utf-8'
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
    with st.expander("🔍 Первые 10 строк"):
        st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    # Оставляем только неделю и месяц
    freq_map = {'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = col1.selectbox("Периодичность", list(freq_map.keys()), index=1)  # месяц по умолчанию
    freq = freq_map[freq_label]
    cats = ['Все'] + sorted(df['category'].unique())
    selected_cat = col2.selectbox("Категория", cats)
    prods = ['Все'] + (sorted(df[df['category']==selected_cat]['product'].unique()) if selected_cat!='Все' else [])
    selected_prod = col3.selectbox("Товар", prods) if prods else None
    horizon = st.slider("Горизонт (периодов)", 1, 52, 5)
    show_advanced = st.checkbox("📊 Расширенная аналитика (декомпозиция, остатки, сравнение)")

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

            # Прогноз total
            res_total = process_target(df_f, 'total', freq, horizon, show_advanced)
            if res_total is None:
                st.error("Недостаточно данных для прогноза total"); st.stop()

            status.text("Прогноз количества..."); progress.progress(50)
            # Прогноз quantity
            res_qty = process_target(df_f, 'quantity', freq, horizon, show_advanced)
            if res_qty is None:
                st.warning("Не удалось построить прогноз количества, будет показан только total")
                res_qty = None

            progress.progress(90)
            status.text("Формирование результатов...")

            # ---------- Вывод результатов для total ----------
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

            # Таблица прогнозов (сумма + количество)
            st.subheader("📋 Прогнозные значения")
            table_df = pd.DataFrame({
                'Дата': res_total['future'].strftime('%d-%m-%Y'),
                'Прогноз суммы': res_total['forecast'].round(2),
                'Нижняя граница (90%)': res_total['lower'].round(2),
                'Верхняя граница (90%)': res_total['upper'].round(2)
            })
            if res_qty is not None:
                table_df['Прогноз количества'] = res_qty['forecast'].round(0).astype(int)
                # Добавляем также нижнюю/верхнюю границу количества? Можно не добавлять, чтобы не перегружать
                st.caption("Широкий доверительный интервал для суммы объясняется волатильностью данных: чем сильнее колебания продаж, тем больше неопределённость прогноза.")
            st.dataframe(table_df, use_container_width=True)

            if show_advanced:
                st.subheader("📊 Расширенная аналитика (сумма продаж)")
                comp = pd.DataFrame([
                    {'Модель':n, 'RMSE':d['rmse'], 'MAPE':d['mape']} for n,d in res_total['models'].items()
                ]).sort_values('MAPE')
                st.dataframe(comp, use_container_width=True)
                fig_c = make_subplots(rows=1, cols=2, subplot_titles=('RMSE','MAPE'))
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['RMSE'], name='RMSE'), 1, 1)
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['MAPE'], name='MAPE'), 1, 2)
                fig_c.update_layout(showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)

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

                resid = res_total['test'].values - res_total['models'][res_total['best_name']]['pred_test']
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=res_total['test'].index, y=resid, name='Остатки'))
                fig_r.add_hline(y=0, line_dash='dash', line_color='red')
                fig_r.update_layout(title='Остатки модели (факт – прогноз) на тестовом периоде')
                st.plotly_chart(fig_r, use_container_width=True)
                st.caption("Остатки показывают разницу между реальными значениями и прогнозом. Если они случайно разбросаны вокруг нуля — модель хорошая.")

            st.caption(f"⏱️ Прогноз построен за {time.time()-start:.1f} сек.")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
        finally:
            del df_f
            gc.collect()
