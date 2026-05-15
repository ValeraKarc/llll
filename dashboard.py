import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, time, gc, warnings, hashlib
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

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

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ===================== HOLIDAYS =====================
HOLIDAYS = {(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,23),(3,8),(5,1),(5,9),(6,12),(11,4)}

def is_holiday(dt): return (dt.month, dt.day) in HOLIDAYS

# ===================== METRICS =====================
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else np.inf

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100 if np.sum(mask) > 0 else np.inf

def mae(y_true, y_pred): return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# ===================== DATA CLEANING =====================
def remove_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    clean = series.copy()
    clean[(clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)] = np.nan
    return clean.interpolate().bfill().ffill()

def validate_csv(df):
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.startswith(('=','+','-','@')).any():
            return False, f"Опасные ячейки в '{col}'"
    return True, "OK"

# ===================== ENCODING =====================
def detect_encoding(raw_bytes):
    encodings = ['utf-8-sig','utf-8','cp1251','windows-1251','iso-8859-5','koi8-r','cp866','latin-1']
    try:
        import chardet
        d = chardet.detect(raw_bytes)
        if d and d.get('confidence',0) > 0.6:
            e = d.get('encoding','').lower()
            mapping = {'windows-1251':'cp1251','ascii':'utf-8'}
            enc = mapping.get(e,e)
            if enc and enc not in encodings: encodings.insert(0, enc)
            elif enc in encodings: encodings.remove(enc); encodings.insert(0, enc)
    except ImportError: pass

    for enc in encodings:
        try:
            decoded = raw_bytes.decode(enc)
            if '\ufffd' not in decoded: return enc, decoded
        except: pass
    return 'latin-1', raw_bytes.decode('latin-1')

# ===================== BASELINE MODELS =====================
def naive_fc(train, h): return np.full(h, train.iloc[-1] if len(train) > 0 else 0)

def seasonal_naive_fc(train, h, sp):
    if len(train) >= sp:
        return np.tile(train.iloc[-sp:].values, (h // sp) + 1)[:h]
    return naive_fc(train, h)

# ===================== ML TRAINING =====================
def make_features(series, lags, holidays=None):
    X = pd.DataFrame(index=series.index)
    for lag in range(1, lags+1): X[f'lag_{lag}'] = series.shift(lag)
    X['month'] = series.index.month
    X['dayofweek'] = series.index.dayofweek
    X['quarter'] = series.index.quarter
    if holidays is not None: X['holiday'] = holidays
    return X

def train_ml(model, train_series, test_idx, lags, freq, holidays=None):
    X = make_features(train_series, lags, holidays)
    y = train_series.copy()
    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < max(lags*2, 10): return None, None, None

    model.fit(X, y)
    preds, hist = [], y.iloc[-lags:].tolist()

    for dt in test_idx:
        feat = {f'lag_{j+1}': hist[-j-1] if len(hist) > j else np.nan for j in range(lags)}
        feat.update({'month': dt.month, 'dayofweek': dt.dayofweek, 'quarter': dt.quarter})
        if holidays is not None: feat['holiday'] = 1 if is_holiday(dt) else 0
        X_row = pd.DataFrame([feat])[X.columns]
        pred = model.predict(X_row)[0]
        preds.append(pred)
        hist.append(pred)
        if len(hist) > lags: hist.pop(0)

    return np.array(preds), model, X

# ===================== FORECASTING =====================
def process_target(df_f, target_col, freq, horizon):
    ts = df_f.set_index('datetime')[target_col].astype(np.float64).resample(freq).sum()
    ts = ts.interpolate().bfill().ffill().dropna()
    if len(ts) < horizon + 10: return None

    ts = remove_outliers(ts)
    train_size = max(int(len(ts)*0.8), len(ts)-horizon)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]
    if len(test) == 0:
        test = ts.iloc[-horizon:]
        train = ts.iloc[:-horizon]

    sp = {'W-MON':52,'MS':12}[freq]
    if sp >= len(train): sp = max(2, len(train)//3)
    lags = min({'W-MON':24,'MS':6}[freq], len(train)//3)

    holidays = None
    if freq in ('D','W-MON'):
        holidays = pd.Series([1 if is_holiday(d) else 0 for d in train.index], index=train.index, dtype=np.int8)

    models = {}

    # Naive
    n_pred = naive_fc(train, len(test))
    models['Naive'] = {'rmse': np.sqrt(mean_squared_error(test,n_pred)), 'mape': mape(test,n_pred),
                       'smape': smape(test,n_pred), 'mae': mae(test,n_pred), 'pred': n_pred, 'model': None, 'X': None}

    # Seasonal Naive
    if len(train) >= sp:
        sn_pred = seasonal_naive_fc(train, len(test), sp)
        models['Seasonal Naive'] = {'rmse': np.sqrt(mean_squared_error(test,sn_pred)), 'mape': mape(test,sn_pred),
                                    'smape': smape(test,sn_pred), 'mae': mae(test,sn_pred), 'pred': sn_pred, 'model': None, 'X': None}

    # Holt-Winters
    try:
        if len(train) >= 2*sp + 5:
            hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp, initialization_method='estimated').fit()
            hw_pred = hw.forecast(len(test))
            models['Holt-Winters'] = {'rmse': np.sqrt(mean_squared_error(test,hw_pred)), 'mape': mape(test,hw_pred),
                                      'smape': smape(test,hw_pred), 'mae': mae(test,hw_pred), 'pred': hw_pred, 'model': hw, 'X': None}
    except Exception as e: st.warning(f"Holt-Winters: {str(e)[:80]}")

    # Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        rf_pred, rf_m, X_rf = train_ml(rf, train, test.index, lags, freq, holidays)
        if rf_pred is not None:
            models['Random Forest'] = {'rmse': np.sqrt(mean_squared_error(test,rf_pred)), 'mape': mape(test,rf_pred),
                                       'smape': smape(test,rf_pred), 'mae': mae(test,rf_pred), 'pred': rf_pred, 'model': rf_m, 'X': X_rf}
    except Exception as e: st.warning(f"Random Forest: {str(e)[:80]}")

    # XGBoost
    if HAS_XGB:
        try:
            xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
            xgb_pred, xgb_m, X_xgb = train_ml(xgb, train, test.index, lags, freq, holidays)
            if xgb_pred is not None:
                models['XGBoost'] = {'rmse': np.sqrt(mean_squared_error(test,xgb_pred)), 'mape': mape(test,xgb_pred),
                                     'smape': smape(test,xgb_pred), 'mae': mae(test,xgb_pred), 'pred': xgb_pred, 'model': xgb_m, 'X': X_xgb}
        except Exception as e: st.warning(f"XGBoost: {str(e)[:80]}")

    if not models: return None

    best_name = min(models, key=lambda k: models[k]['mape'] if models[k]['mape'] != np.inf else models[k]['rmse']*100)
    best = models[best_name]

    if best['mape'] > models['Naive']['mape'] * 1.5 and models['Naive']['mape'] != np.inf:
        st.info(f"⚠️ Модель {best_name} незначительно лучше наивного прогноза")

    full_ts = pd.concat([train, test])

    # CRITICAL FIX: Generate future dates using pure Python, zero pandas arithmetic
    # Get the last timestamp as plain Python datetime
    last_idx = full_ts.index[-1]

    # Convert to plain Python datetime object (no pandas Timestamp, no freq)
    if hasattr(last_idx, 'to_pydatetime'):
        last_py_dt = last_idx.to_pydatetime()
    else:
        last_py_dt = datetime(last_idx.year, last_idx.month, last_idx.day)

    # Generate future dates as plain Python datetimes first
    future_py_dates = []

    if freq == 'W-MON':
        # Weekly: add 7 days each step
        # Find next Monday
        days_until_monday = (7 - last_py_dt.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7  # If already Monday, go to next Monday
        current = last_py_dt + timedelta(days=days_until_monday)
        for _ in range(horizon):
            future_py_dates.append(current)
            current += timedelta(days=7)
    elif freq == 'MS':
        # Monthly: first day of next month
        yr, mo = last_py_dt.year, last_py_dt.month
        for _ in range(horizon):
            if mo == 12:
                yr, mo = yr + 1, 1
            else:
                mo = mo + 1
            future_py_dates.append(datetime(yr, mo, 1))
    else:
        # Daily fallback
        current = last_py_dt + timedelta(days=1)
        for _ in range(horizon):
            future_py_dates.append(current)
            current += timedelta(days=1)

    # Convert Python datetimes to pandas DatetimeIndex WITHOUT freq
    future = pd.DatetimeIndex([pd.Timestamp(d) for d in future_py_dates])

    # Final forecast
    if best_name == 'Holt-Winters' and best['model'] is not None:
        fc = ExponentialSmoothing(full_ts, trend='add', seasonal='add', seasonal_periods=sp, initialization_method='estimated').fit().forecast(horizon)
    elif best_name in ('Naive','Seasonal Naive'):
        fc = seasonal_naive_fc(full_ts, horizon, sp) if best_name == 'Seasonal Naive' and len(full_ts) >= sp else naive_fc(full_ts, horizon)
    else:
        X_full = make_features(full_ts, lags, holidays)
        y_full = full_ts.copy()
        valid = ~X_full.isna().any(axis=1)
        X_full, y_full = X_full.loc[valid], y_full.loc[valid]

        if best_name == 'Random Forest':
            m = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        else:
            m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_full, y_full)

        future_hol = [1 if is_holiday(d) else 0 for d in future] if freq in ('D','W-MON') else None
        hist = y_full.iloc[-lags:].tolist()
        fc = []

        for i in range(horizon):
            feat = {f'lag_{j+1}': hist[-j-1] if len(hist) > j else np.nan for j in range(lags)}
            feat.update({'month': future[i].month, 'dayofweek': future[i].dayofweek, 'quarter': future[i].quarter})
            if future_hol: feat['holiday'] = future_hol[i]
            X_row = pd.DataFrame([feat])[X_full.columns]
            p = m.predict(X_row)[0]
            fc.append(p)
            hist.append(p)
            if len(hist) > lags: hist.pop(0)
        fc = np.array(fc)

    residuals = np.array(test) - np.array(best['pred'])
    if len(residuals) > 10:
        lower = fc + np.percentile(residuals, 5)
        upper = fc + np.percentile(residuals, 95)
    else:
        std = np.std(residuals)
        lower, upper = fc - 1.645*std, fc + 1.645*std

    if target_col == 'quantity':
        lower = np.maximum(lower, 0)
        fc = np.maximum(fc, 0)

    return {'train': train, 'test': test, 'future': future, 'forecast': fc, 'lower': lower, 'upper': upper,
            'rmse': best['rmse'], 'mape': best['mape'], 'smape': best['smape'], 'mae': best['mae'],
            'best_name': best_name, 'models': models, 'sp': sp, 'lags': lags, 'freq': freq,
            'X_best': best.get('X', None), 'residuals': residuals}

# ===================== PDF REPORT =====================
def generate_pdf(res_total, res_qty, horizon, freq_label):
    if not HAS_REPORTLAB: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Отчет о прогнозировании продаж", styles['Heading1']), Spacer(1,20)]

    info = [['Параметр','Значение'],
            ['Дата', datetime.now().strftime('%d-%m-%Y %H:%M')],
            ['Периодичность', freq_label], ['Горизонт', str(horizon)],
            ['Модель', res_total['best_name']], ['MAPE', f"{res_total['mape']:.2f}%"], ['RMSE', f"{res_total['rmse']:,.2f}"]]
    if res_qty: info.extend([['Модель (кол-во)', res_qty['best_name']], ['MAPE (кол-во)', f"{res_qty['mape']:.2f}%"]])

    t = Table(info, colWidths=[250,250])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                          ('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.extend([t, Spacer(1,20), Paragraph("Прогноз", styles['Heading2']), Spacer(1,10)])

    data = [['Дата','Прогноз суммы','Нижняя','Верхняя']]
    if res_qty: data[0].append('Прогноз кол-ва')
    for i in range(len(res_total['future'])):
        row = [res_total['future'][i].strftime('%d-%m-%Y'), f"{res_total['forecast'][i]:,.2f}",
               f"{res_total['lower'][i]:,.2f}", f"{res_total['upper'][i]:,.2f}"]
        if res_qty: row.append(f"{res_qty['forecast'][i]:,.0f}")
        data.append(row)

    t2 = Table(data, colWidths=[120]*5 if res_qty else [150]*4)
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                            ('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.append(t2)
    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

# ===================== UI =====================
st.set_page_config(page_title="Прогнозирование продаж", layout="wide")
st.title("📈 Интеллектуальная модель прогнозирования продаж")
st.markdown("Загрузите CSV-файл с продажами и получите прогноз с автовыбором лучшей модели.")

with st.sidebar:
    st.info("**Столбцы:** date, time, category, product, quantity, price, total\n**Кодировка:** авто\n**Макс. размер:** 150 МБ")
    if st.button("🗑️ Очистить сессию"):
        for k in list(st.session_state.keys()):
            if k not in ['session_mgr','upload_key']: del st.session_state[k]
        st.session_state.upload_key = st.session_state.get('upload_key',0) + 1
        gc.collect()
        st.success("Сессия очищена")
        st.rerun()

if 'session_mgr' not in st.session_state:
    st.session_state.session_mgr = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    st.session_state.upload_key = 0

uploaded = st.file_uploader("📂 Загрузите CSV (до 150 МБ)", type="csv", key=f"uploader_{st.session_state.upload_key}")

if uploaded:
    if uploaded.size > 150*1024*1024:
        st.error("❌ Файл > 150 МБ"); st.stop()

    try:
        raw = uploaded.read()
        enc, text = detect_encoding(raw)
        st.info(f"🔤 Кодировка: **{enc}**")

        df = pd.read_csv(io.StringIO(text), dtype=str)
        df.columns = df.columns.str.strip().str.lower()

        ok, msg = validate_csv(df)
        if not ok: st.error(f"⚠️ {msg}"); st.stop()

        required = ['date','time','category','product','quantity','price','total']
        missing = [c for c in required if c not in df.columns]
        if missing: st.error(f"❌ Нет столбцов: {', '.join(missing)}"); st.stop()

        for c in ['quantity','price','total']: df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in ['date','time','category','product']: df[c] = df[c].astype(str).str.strip()

        df['time'] = df['time'].replace(['','nan','None','null'], '')
        time_empty = df['time'].eq('').all()
        df['datetime'] = pd.to_datetime(df['date'] + (' ' + df['time'].fillna('').replace('','00:00:00') if not time_empty else ''), errors='coerce')

        df.dropna(subset=['datetime','quantity','price','total'], inplace=True)
        df = df[df['total'].abs() > 0].drop_duplicates(subset=['datetime','category','product','quantity','price'])
        df.sort_values('datetime', inplace=True)

        if df.empty: st.error("❌ Нет данных после очистки"); st.stop()

        st.success(f"✅ {len(df):,} записей | {df['datetime'].min().strftime('%d-%m-%Y')} — {df['datetime'].max().strftime('%d-%m-%Y')}")

        # Show first 5 rows
        with st.expander("🔍 Первые 5 строк", expanded=True):
            preview = df.head(5).copy()
            preview['datetime'] = preview['datetime'].dt.strftime('%d-%m-%Y %H:%M')
            st.dataframe(preview, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Записей", f"{len(df):,}")
        col2.metric("Категорий", df['category'].nunique())
        col3.metric("Товаров", df['product'].nunique())
        col4.metric("Период (дней)", (df['datetime'].max() - df['datetime'].min()).days)

    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.info("Проверьте формат CSV: разделитель — запятая, первая строка — заголовки. Попробуйте сохранить из Excel в UTF-8.")
        st.stop()

    # Forecast settings
    st.markdown("---")
    st.subheader("🔧 Параметры прогноза")

    c1, c2, c3, c4 = st.columns(4)
    freq_map = {'Неделя':'W-MON','Месяц':'MS'}
    freq_label = c1.selectbox("Периодичность", list(freq_map.keys()), index=1)
    freq = freq_map[freq_label]

    cats = ['Все'] + sorted(df['category'].unique().tolist())
    cat = c2.selectbox("Категория", cats)
    prods = ['Все'] + (sorted(df[df['category']==cat]['product'].unique().tolist()) if cat!='Все' else [])
    prod = c3.selectbox("Товар", prods) if prods else None

    max_h = {'W-MON':52,'MS':24}[freq]
    horizon = c4.number_input("Горизонт", min_value=1, max_value=max_h, value=min(12,max_h))

    with st.expander("⚙️ Дополнительно"):
        show_adv = st.checkbox("Расширенная аналитика", False)
        show_ret = st.checkbox("Учитывать возвраты", False)

    df_w = df.copy() if show_ret else df[df['total'] > 0].copy()
    df_f = df_w[df_w['category']==cat] if cat!='Все' else df_w.copy()
    if prod and prod!='Все': df_f = df_f[df_f['product']==prod]

    if df_f.empty: st.warning("Нет данных для фильтров"); st.stop()

    min_req = {'W-MON':10,'MS':6}[freq]
    if len(df_f.set_index('datetime')['total'].resample(freq).sum().dropna()) < min_req:
        st.warning(f"Нужно минимум {min_req} периодов"); st.stop()

    if st.button("🚀 Построить прогноз", type="primary", use_container_width=True):
        start = time.time()
        prog = st.progress(0)
        status = st.empty()

        try:
            status.text("Анализ суммы..."); prog.progress(15)
            res_total = process_target(df_f, 'total', freq, horizon)
            if res_total is None: st.error("Недостаточно данных"); st.stop()

            status.text("Анализ количества..."); prog.progress(50)
            res_qty = process_target(df_f, 'quantity', freq, horizon)

            prog.progress(85); status.text("Формирование отчета...")
            prog.empty(); status.empty()

            # Results
            st.markdown("---")
            st.subheader(f"🏆 {res_total['best_name']}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", f"{res_total['rmse']:,.2f}")
            m2.metric("MAPE", f"{res_total['mape']:.2f}%")
            m3.metric("SMAPE", f"{res_total['smape']:.2f}%")
            m4.metric("MAE", f"{res_total['mae']:,.2f}")

            others = [m for m in res_total['models'] if m != res_total['best_name']]
            if others:
                alt = min(others, key=lambda x: res_total['models'][x]['mape'] if res_total['models'][x]['mape']!=np.inf else 999999)
                st.caption(f"📌 Альтернатива: {alt} (MAPE: {res_total['models'][alt]['mape']:.2f}%)")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_total['train'].index, y=res_total['train'].values, name='Обучение', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=res_total['test'].index, y=res_total['test'].values, name='Тест', line=dict(color='#ff7f0e')))
            fig.add_trace(go.Scatter(x=res_total['future'], y=res_total['forecast'], name='Прогноз', line=dict(color='#2ca02c', width=3), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=np.concatenate([res_total['future'],res_total['future'][::-1]]),
                                     y=np.concatenate([res_total['upper'],res_total['lower'][::-1]]),
                                     fill='toself', fillcolor='rgba(44,160,44,0.15)', line=dict(color='rgba(255,255,255,0)'), name='90% CI', hoverinfo='skip'))
            split = res_total['test'].index[0] if len(res_total['test']) > 0 else res_total['train'].index[-1]
            fig.add_vline(x=split, line=dict(color='red', dash='dash'), annotation_text="Прогноз", annotation_position="top")
            fig.update_layout(title=f'Прогноз суммы — {res_total["best_name"]}', xaxis_title='Дата', yaxis_title='Сумма', hovermode='x unified', height=500)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # Table
            st.subheader("📋 Прогноз")
            tbl = {'Дата': res_total['future'].strftime('%d-%m-%Y'), 'Прогноз суммы': res_total['forecast'].round(2),
                   'Нижняя (90%)': res_total['lower'].round(2), 'Верхняя (90%)': res_total['upper'].round(2)}
            if res_qty:
                tbl['Прогноз кол-ва'] = res_qty['forecast'].round(0).astype(int)
                tbl['Нижняя кол-ва'] = np.maximum(res_qty['lower'],0).round(0).astype(int)
                tbl['Верхняя кол-ва'] = res_qty['upper'].round(0).astype(int)

            tbl_df = pd.DataFrame(tbl)
            st.dataframe(tbl_df, use_container_width=True)

            # Downloads
            csv_buf = io.StringIO()
            tbl_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            st.download_button("📥 Скачать CSV", csv_buf.getvalue(), f"forecast_{st.session_state.session_mgr}.csv", "text/csv")

            if HAS_REPORTLAB:
                pdf = generate_pdf(res_total, res_qty, horizon, freq_label)
                if pdf: st.download_button("📄 Скачать PDF", pdf, f"report_{st.session_state.session_mgr}.pdf", "application/pdf")

            # Advanced analytics
            if show_adv:
                st.markdown("---")
                st.subheader("📊 Расширенная аналитика")

                with st.expander("📈 Сравнение моделей"):
                    comp = pd.DataFrame([{'Модель':n,'RMSE':d['rmse'],'MAPE':d['mape'],'SMAPE':d['smape'],'MAE':d['mae']} for n,d in res_total['models'].items()]).sort_values('MAPE')
                    st.dataframe(comp, use_container_width=True)
                    fc = make_subplots(rows=2, cols=2, subplot_titles=('RMSE','MAPE','SMAPE','MAE'), vertical_spacing=0.15)
                    fc.add_trace(go.Bar(x=comp['Модель'], y=comp['RMSE']), 1,1)
                    fc.add_trace(go.Bar(x=comp['Модель'], y=comp['MAPE']), 1,2)
                    fc.add_trace(go.Bar(x=comp['Модель'], y=comp['SMAPE']), 2,1)
                    fc.add_trace(go.Bar(x=comp['Модель'], y=comp['MAE']), 2,2)
                    fc.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fc, use_container_width=True)

                if len(res_total['train']) >= 2*res_total['sp'] + 5:
                    with st.expander("🔄 Декомпозиция"):
                        try:
                            dec = seasonal_decompose(res_total['train'], model='additive', period=res_total['sp'])
                            fd = make_subplots(rows=4, cols=1, subplot_titles=('Наблюдения','Тренд','Сезонность','Остатки'), vertical_spacing=0.08)
                            fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.observed), 1,1)
                            fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.trend), 2,1)
                            fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.seasonal), 3,1)
                            fd.add_trace(go.Scatter(x=res_total['train'].index, y=dec.resid), 4,1)
                            fd.update_layout(height=900, showlegend=False)
                            st.plotly_chart(fd, use_container_width=True)
                        except: pass

                with st.expander("📉 Остатки"):
                    resid = res_total['residuals']
                    r1, r2 = st.columns(2)
                    fr = go.Figure(go.Scatter(x=res_total['test'].index, y=resid, mode='markers', marker=dict(size=8)))
                    fr.add_hline(y=0, line_dash='dash', line_color='red')
                    fr.update_layout(title='Остатки', xaxis_title='Дата', yaxis_title='Ошибка')
                    r1.plotly_chart(fr, use_container_width=True)

                    fh = go.Figure(go.Histogram(x=resid, nbinsx=20))
                    fh.update_layout(title='Распределение остатков', xaxis_title='Ошибка')
                    r2.plotly_chart(fh, use_container_width=True)

                    if len(resid) > 5:
                        av = acf(resid, nlags=min(20, len(resid)//2))
                        fa = go.Figure()
                        for i,v in enumerate(av): fa.add_vline(x=i, line_width=3, line_color='blue', opacity=abs(v))
                        cl = 1.96/np.sqrt(len(resid))
                        fa.add_hline(y=cl, line_dash='dash', line_color='red')
                        fa.add_hline(y=-cl, line_dash='dash', line_color='red')
                        fa.add_hline(y=0, line_color='black')
                        fa.update_layout(title='ACF остатков', xaxis_title='Лаг', yaxis_title='ACF')
                        st.plotly_chart(fa, use_container_width=True)

                if res_total['best_name'] not in ('Holt-Winters','Naive','Seasonal Naive') and res_total['X_best'] is not None:
                    if hasattr(res_total['models'][res_total['best_name']]['model'], 'feature_importances_'):
                        with st.expander("🔍 Важность признаков"):
                            imp = res_total['models'][res_total['best_name']]['model'].feature_importances_
                            fi = pd.DataFrame({'Признак': res_total['X_best'].columns, 'Важность': imp}).sort_values('Важность', ascending=True)
                            fim = go.Figure(go.Bar(x=fi['Важность'], y=fi['Признак'], orientation='h'))
                            fim.update_layout(title='Важность признаков', xaxis_title='Важность', height=400)
                            st.plotly_chart(fim, use_container_width=True)

                if res_total['X_best'] is not None:
                    with st.expander("🔗 Корреляции"):
                        Xc = res_total['X_best'].copy()
                        Xc['target'] = res_total['train'].loc[Xc.index]
                        cr = Xc.corr()
                        fcr = go.Figure(go.Heatmap(z=cr.values, x=cr.columns, y=cr.index, colorscale='RdBu_r', zmin=-1, zmax=1,
                                                   text=np.round(cr.values,2), texttemplate='%{text}', textfont={"size":10}))
                        fcr.update_layout(title='Корреляционная матрица', height=500)
                        st.plotly_chart(fcr, use_container_width=True)

            st.caption(f"⏱️ За {time.time()-start:.1f} сек | Сессия: {st.session_state.session_mgr}")

        except Exception as e:
            prog.empty(); status.empty()
            st.error(f"❌ Ошибка: {str(e)}")
            st.info("Рекомендации: проверьте данные, попробуйте другую периодичность или уменьшите горизонт.")
        finally:
            if 'df_f' in locals(): del df_f
            gc.collect()

st.markdown("---")
st.caption("🔒 Данные обрабатываются только в RAM и удаляются после сессии. Логирование отключено.")
