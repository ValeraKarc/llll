import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64, time, gc, os, warnings, io, threading
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # No display needed
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

try:
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# =============================================================================
# SESSION MANAGEMENT & SECURITY
# =============================================================================

class SessionManager:
    """Manages session lifecycle and data cleanup."""

    def __init__(self):
        self._session_id = hashlib.sha256(
            f"{time.time()}-{threading.current_thread().ident}".encode()
        ).hexdigest()[:16]
        self._data_store = {}
        self._creation_time = time.time()

    @property
    def session_id(self):
        return self._session_id

    def store(self, key, value):
        """Store data in session memory only."""
        self._data_store[key] = value

    def get(self, key, default=None):
        return self._data_store.get(key, default)

    def clear(self):
        """Irreversibly delete all session data."""
        for key in list(self._data_store.keys()):
            if isinstance(self._data_store[key], pd.DataFrame):
                del self._data_store[key]
            else:
                self._data_store[key] = None
        self._data_store.clear()
        gc.collect()

    def is_expired(self, timeout_minutes=30):
        return (time.time() - self._creation_time) > (timeout_minutes * 60)

# Initialize session manager in Streamlit state
if 'session_mgr' not in st.session_state:
    st.session_state.session_mgr = SessionManager()
    st.session_state.upload_key = 0

session_mgr = st.session_state.session_mgr

# =============================================================================
# RUSSIAN HOLIDAYS
# =============================================================================

HOLIDAY_DATES = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}

def is_holiday(dt):
    """Check if date is a Russian holiday."""
    return (dt.month, dt.day) in HOLIDAY_DATES

# =============================================================================
# METRICS
# =============================================================================

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    """Symmetric MAPE - more robust for values near zero."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# =============================================================================
# DATA CLEANING
# =============================================================================

def remove_outliers(series, method='iqr', threshold=1.5):
    """Remove outliers using IQR or Z-score method."""
    clean = series.copy()

    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    else:  # z-score
        mean = series.mean()
        std = series.std()
        lower = mean - 3 * std
        upper = mean + 3 * std

    clean[(clean < lower) | (clean > upper)] = np.nan

    # Interpolate with method selection based on data size
    if len(clean) > 100:
        clean = clean.interpolate(method='linear')
    else:
        clean = clean.interpolate(method='polynomial', order=2)

    clean = clean.bfill().ffill()
    return clean

def validate_csv_security(df):
    """Check for CSV injection attacks."""
    dangerous_prefixes = ('=', '+', '-', '@', '\t', '\r')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.startswith(dangerous_prefixes).any():
            suspicious = df[col][df[col].astype(str).str.startswith(dangerous_prefixes)].iloc[:3]
            return False, f"Обнаружены потенциально опасные ячейки в столбце '{col}': {suspicious.tolist()[:3]}"

    return True, "OK"

# =============================================================================
# NAIVE BASELINE MODEL
# =============================================================================

def naive_forecast(train, horizon):
    """Naive baseline: last value repeated."""
    last_value = train.iloc[-1] if len(train) > 0 else 0
    return np.full(horizon, last_value)

def seasonal_naive_forecast(train, horizon, seasonal_period):
    """Seasonal naive: repeat last seasonal period."""
    if len(train) >= seasonal_period:
        seasonal_values = train.iloc[-seasonal_period:].values
        repeats = (horizon // seasonal_period) + 1
        forecast = np.tile(seasonal_values, repeats)[:horizon]
        return forecast
    return naive_forecast(train, horizon)

# =============================================================================
# ML MODEL TRAINING
# =============================================================================

def create_lag_features(series, lags, holiday_series=None):
    """Create lag features for ML models."""
    X = pd.DataFrame(index=series.index)
    for lag in range(1, lags + 1):
        X[f'lag_{lag}'] = series.shift(lag)

    # Add time-based features
    X['month'] = series.index.month
    X['dayofweek'] = series.index.dayofweek
    X['quarter'] = series.index.quarter

    if holiday_series is not None:
        X['holiday'] = holiday_series

    return X

def train_ml_model(model, train_series, test_index, lags, freq, holiday_series=None):
    """Train ML model with recursive multi-step forecasting."""
    X = create_lag_features(train_series, lags, holiday_series)
    y = train_series.copy()

    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    if len(X) < max(lags * 2, 10):
        return None, None, None

    model.fit(X, y)

    # Recursive forecasting
    test_pred = []
    hist = y.iloc[-lags:].tolist()

    for i, dt in enumerate(test_index):
        feat = {}
        for j in range(lags):
            feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan

        feat['month'] = dt.month
        feat['dayofweek'] = dt.dayofweek
        feat['quarter'] = dt.quarter

        if holiday_series is not None:
            feat['holiday'] = 1 if is_holiday(dt) else 0

        X_row = pd.DataFrame([feat])
        # Ensure column order matches training
        X_row = X_row[X.columns]

        pred = model.predict(X_row)[0]
        test_pred.append(pred)

        # Maintain fixed-size history
        hist.append(pred)
        if len(hist) > lags:
            hist.pop(0)

    return np.array(test_pred), model, X

# =============================================================================
# MAIN FORECASTING PIPELINE
# =============================================================================

def get_seasonal_period(freq):
    """Get seasonal period based on frequency."""
    periods = {
        'H': 24,      # Hourly: daily seasonality
        'D': 7,       # Daily: weekly seasonality
        'W-MON': 52,  # Weekly: yearly seasonality
        'MS': 12,     # Monthly: yearly seasonality
    }
    return periods.get(freq, 12)

def get_lags(freq, train_length):
    """Determine optimal number of lags."""
    base_lags = {
        'H': 48,      # 2 days
        'D': 14,      # 2 weeks
        'W-MON': 24,  # ~6 months
        'MS': 6,      # 6 months
    }
    lags = base_lags.get(freq, 6)
    return min(lags, train_length // 3)

def process_target(df_f, target_col, freq, horizon):
    """Process single target column and return forecast results."""

    # Aggregate data
    ts = df_f.set_index('datetime')[target_col].astype(np.float64)
    ts = ts.resample(freq).sum()
    ts = ts.interpolate().bfill().ffill().dropna()

    if len(ts) < horizon + 10:
        return None

    # Clean outliers
    ts = remove_outliers(ts)

    # Split data
    train_size = max(int(len(ts) * 0.8), len(ts) - horizon)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]

    if len(test) == 0:
        test = ts.iloc[-horizon:]
        train = ts.iloc[:-horizon]

    # Parameters
    sp = get_seasonal_period(freq)
    if sp >= len(train):
        sp = max(2, len(train) // 3)

    lags = get_lags(freq, len(train))

    # Holiday features
    holiday_series = None
    if freq in ('D', 'W-MON'):
        holiday_series = pd.Series(
            [1 if is_holiday(d) else 0 for d in train.index],
            index=train.index, dtype=np.int8
        )

    models_results = {}

    # 1. NAIVE BASELINE
    naive_pred = naive_forecast(train, len(test))
    models_results['Naive'] = {
        'rmse': np.sqrt(mean_squared_error(test, naive_pred)),
        'mape': mape(test, naive_pred),
        'smape': smape(test, naive_pred),
        'mae': mae(test, naive_pred),
        'pred_test': naive_pred,
        'model': None,
        'X_train': None
    }

    # 2. SEASONAL NAIVE
    if len(train) >= sp:
        seasonal_pred = seasonal_naive_forecast(train, len(test), sp)
        models_results['Seasonal Naive'] = {
            'rmse': np.sqrt(mean_squared_error(test, seasonal_pred)),
            'mape': mape(test, seasonal_pred),
            'smape': smape(test, seasonal_pred),
            'mae': mae(test, seasonal_pred),
            'pred_test': seasonal_pred,
            'model': None,
            'X_train': None
        }

    # 3. HOLT-WINTERS
    try:
        if len(train) >= 2 * sp + 5:
            hw = ExponentialSmoothing(
                train, 
                trend='add', 
                seasonal='add',
                seasonal_periods=sp,
                initialization_method='estimated'
            ).fit()
            pred_hw = hw.forecast(len(test))

            models_results['Holt-Winters'] = {
                'rmse': np.sqrt(mean_squared_error(test, pred_hw)),
                'mape': mape(test, pred_hw),
                'smape': smape(test, pred_hw),
                'mae': mae(test, pred_hw),
                'pred_test': pred_hw,
                'model': hw,
                'X_train': None
            }
    except Exception as e:
        st.warning(f"Holt-Winters не удалось обучить ({target_col}): {str(e)[:100]}")

    # 4. RANDOM FOREST
    try:
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        )
        pred_rf, rf_model, X_rf = train_ml_model(rf, train, test.index, lags, freq, holiday_series)

        if pred_rf is not None:
            models_results['Random Forest'] = {
                'rmse': np.sqrt(mean_squared_error(test, pred_rf)),
                'mape': mape(test, pred_rf),
                'smape': smape(test, pred_rf),
                'mae': mae(test, pred_rf),
                'pred_test': pred_rf,
                'model': rf_model,
                'X_train': X_rf
            }
    except Exception as e:
        st.warning(f"Random Forest ошибка ({target_col}): {str(e)[:100]}")

    # 5. XGBOOST
    if HAS_XGB:
        try:
            xgb = XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42, 
                verbosity=0, 
                n_jobs=-1
            )
            pred_xgb, xgb_model, X_xgb = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)

            if pred_xgb is not None:
                models_results['XGBoost'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred_xgb)),
                    'mape': mape(test, pred_xgb),
                    'smape': smape(test, pred_xgb),
                    'mae': mae(test, pred_xgb),
                    'pred_test': pred_xgb,
                    'model': xgb_model,
                    'X_train': X_xgb
                }
        except Exception as e:
            st.warning(f"XGBoost ошибка ({target_col}): {str(e)[:100]}")

    if not models_results:
        return None

    # Select best model by MAPE (with fallback to RMSE if MAPE is inf)
    def model_score(name):
        mape_val = models_results[name]['mape']
        return mape_val if mape_val != np.inf else models_results[name]['rmse'] * 100

    best_name = min(models_results, key=model_score)
    best = models_results[best_name]

    # Check if best model is better than naive
    naive_mape = models_results['Naive']['mape']
    if best['mape'] > naive_mape * 1.5 and naive_mape != np.inf:
        st.info(f"⚠️ Выбранная модель ({best_name}) незначительно лучше наивного прогноза. Рекомендуется проверить данные.")

    # Final forecast on full dataset
    full_ts = pd.concat([train, test])

    # Generate future dates
    freq_offsets = {
        'H': pd.DateOffset(hours=1),
        'D': pd.DateOffset(days=1),
        'W-MON': pd.DateOffset(weeks=1),
        'MS': pd.DateOffset(months=1),
    }
    start_future = full_ts.index[-1] + freq_offsets.get(freq, pd.DateOffset(days=1))
    future = pd.date_range(start=start_future, periods=horizon, freq=freq)

    # Generate final forecast
    if best_name == 'Holt-Winters' and best['model'] is not None:
        full_model = ExponentialSmoothing(
            full_ts, 
            trend='add', 
            seasonal='add',
            seasonal_periods=sp,
            initialization_method='estimated'
        ).fit()
        forecast = full_model.forecast(horizon)

    elif best_name in ('Naive', 'Seasonal Naive'):
        if best_name == 'Seasonal Naive' and len(full_ts) >= sp:
            forecast = seasonal_naive_forecast(full_ts, horizon, sp)
        else:
            forecast = naive_forecast(full_ts, horizon)

    else:
        # ML models - retrain on full data
        X_full = create_lag_features(full_ts, lags, holiday_series)
        y_full = full_ts.copy()
        valid = ~X_full.isna().any(axis=1)
        X_full, y_full = X_full.loc[valid], y_full.loc[valid]

        if best_name == 'Random Forest':
            full_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            )
        else:
            full_model = XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42, 
                verbosity=0, 
                n_jobs=-1
            )

        full_model.fit(X_full, y_full)

        # Forecast
        future_hol = None
        if freq in ('D', 'W-MON'):
            future_hol = [1 if is_holiday(d) else 0 for d in future]

        hist = y_full.iloc[-lags:].tolist()
        forecast = []

        for i in range(horizon):
            feat = {}
            for j in range(lags):
                feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan

            feat['month'] = future[i].month
            feat['dayofweek'] = future[i].dayofweek
            feat['quarter'] = future[i].quarter

            if future_hol is not None:
                feat['holiday'] = future_hol[i]

            X_row = pd.DataFrame([feat])
            X_row = X_row[X_full.columns]

            pred = full_model.predict(X_row)[0]
            forecast.append(pred)

            hist.append(pred)
            if len(hist) > lags:
                hist.pop(0)

        forecast = np.array(forecast)

    # Confidence intervals
    residuals = np.array(test) - np.array(best['pred_test'])
    std_res = np.std(residuals)

    # Use empirical quantiles for better intervals
    if len(residuals) > 10:
        lower_q = np.percentile(residuals, 5)
        upper_q = np.percentile(residuals, 95)
        lower = forecast + lower_q
        upper = forecast + upper_q
    else:
        lower = forecast - 1.645 * std_res
        upper = forecast + 1.645 * std_res

    # Ensure non-negative for quantity
    if target_col == 'quantity':
        lower = np.maximum(lower, 0)
        forecast = np.maximum(forecast, 0)

    return {
        'train': train,
        'test': test,
        'future': future,
        'forecast': forecast,
        'lower': lower,
        'upper': upper,
        'rmse': best['rmse'],
        'mape': best['mape'],
        'smape': best['smape'],
        'mae': best['mae'],
        'best_name': best_name,
        'models': models_results,
        'sp': sp,
        'lags': lags,
        'freq': freq,
        'X_train_for_best': best.get('X_train', None),
        'residuals': residuals
    }

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(res_total, res_qty, df_info, horizon, freq_label):
    """Generate PDF report with forecast results."""
    if not HAS_REPORTLAB:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Heading1']
    title_style.fontName = 'DejaVuSans'  # Will use default if not available
    story.append(Paragraph("Отчет о прогнозировании продаж", title_style))
    story.append(Spacer(1, 20))

    # Info
    info_data = [
        ['Параметр', 'Значение'],
        ['Дата формирования', datetime.now().strftime('%d-%m-%Y %H:%M')],
        ['Периодичность', freq_label],
        ['Горизонт прогноза', str(horizon)],
        ['Лучшая модель (сумма)', res_total['best_name']],
        ['MAPE (сумма)', f"{res_total['mape']:.2f}%"],
        ['RMSE (сумма)', f"{res_total['rmse']:,.2f}"],
    ]

    if res_qty:
        info_data.append(['Лучшая модель (кол-во)', res_qty['best_name']])
        info_data.append(['MAPE (кол-во)', f"{res_qty['mape']:.2f}%"])

    info_table = Table(info_data, colWidths=[250, 250])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))

    # Forecast table
    story.append(Paragraph("Прогнозные значения", styles['Heading2']))
    story.append(Spacer(1, 10))

    forecast_data = [['Дата', 'Прогноз суммы', 'Нижняя граница', 'Верхняя граница']]
    if res_qty is not None:
        forecast_data[0].append('Прогноз кол-ва')

    for i in range(len(res_total['future'])):
        row = [
            res_total['future'][i].strftime('%d-%m-%Y'),
            f"{res_total['forecast'][i]:,.2f}",
            f"{res_total['lower'][i]:,.2f}",
            f"{res_total['upper'][i]:,.2f}",
        ]
        if res_qty is not None:
            row.append(f"{res_qty['forecast'][i]:,.0f}")
        forecast_data.append(row)

    forecast_table = Table(forecast_data, colWidths=[120, 120, 120, 120, 120] if res_qty else [150, 150, 150, 150])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(forecast_table)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# =============================================================================
# ENCODING DETECTION - ROBUST VERSION
# =============================================================================

def detect_encoding_robust(raw_bytes):
    """Robust encoding detection with multiple fallbacks."""

    # Priority encodings to try (most common for Russian/Windows files)
    encodings_to_try = [
        'utf-8-sig',      # UTF-8 with BOM (most common for Excel exports)
        'utf-8',          # Standard UTF-8
        'cp1251',         # Windows Cyrillic (very common in Russia)
        'windows-1251',   # Alias for cp1251
        'iso-8859-5',     # ISO Cyrillic
        'koi8-r',         # Legacy Russian encoding
        'cp866',          # DOS Cyrillic
        'latin-1',        # Fallback that never fails (maps bytes 1:1)
    ]

    # Try chardet if available
    try:
        import chardet
        detected = chardet.detect(raw_bytes)
        if detected and detected.get('confidence', 0) > 0.7:
            detected_enc = detected.get('encoding', '').lower()
            # Map common chardet names to Python names
            encoding_map = {
                'windows-1251': 'cp1251',
                'windows-1252': 'cp1252',
                'iso-8859-1': 'latin-1',
                'ascii': 'utf-8',
            }
            mapped_enc = encoding_map.get(detected_enc, detected_enc)
            if mapped_enc:
                # Insert detected encoding at the beginning if not already there
                if mapped_enc not in encodings_to_try:
                    encodings_to_try.insert(0, mapped_enc)
                else:
                    # Move to top
                    encodings_to_try.remove(mapped_enc)
                    encodings_to_try.insert(0, mapped_enc)
    except ImportError:
        pass

    # Try each encoding
    for enc in encodings_to_try:
        try:
            decoded = raw_bytes.decode(enc)
            # Validate: check for common corruption indicators
            # If we see lots of replacement characters or odd patterns, skip
            if '\ufffd' in decoded or '\x00' in decoded[:1000]:
                continue
            return enc, decoded
        except (UnicodeDecodeError, LookupError):
            continue

    # Ultimate fallback: latin-1 never fails (maps bytes 1:1)
    return 'latin-1', raw_bytes.decode('latin-1')

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Интеллектуальная модель прогнозирования продаж",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #145a8a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Интеллектуальная модель прогнозирования продаж</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Загрузите CSV-файл с данными о продажах и получите автоматический прогноз с выбором 
оптимальной модели машинного обучения. Система поддерживает агрегацию по часам, дням, 
неделям и месяцам, учитывает сезонность и праздники РФ.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")

    st.info("""
    **Обязательные столбцы:**
    - `date` — дата продажи
    - `time` — время (можно оставить пустым)
    - `category` — категория товара
    - `product` — наименование товара
    - `quantity` — количество
    - `price` — цена за единицу
    - `total` — общая сумма

    **Поддерживаемые форматы даты:** любой распознаваемый формат
    **Максимальный размер файла:** 150 МБ
    """)

    # Session info
    st.caption(f"Сессия: {session_mgr.session_id}")
    if st.button("🗑️ Очистить сессию", help="Удалить все данные текущей сессии"):
        session_mgr.clear()
        st.session_state.upload_key += 1
        st.success("✅ Данные сессии удалены")
        st.rerun()

# File upload
uploaded = st.file_uploader(
    "📂 Загрузите CSV-файл с данными о продажах",
    type="csv",
    key=f"uploader_{st.session_state.upload_key}",
    help="Максимальный размер: 150 МБ. Данные хранятся только в оперативной памяти."
)

if uploaded:
    # Security: file size check
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Файл превышает максимальный размер 150 МБ. Пожалуйста, разделите данные или сожмите файл.")
        st.stop()

    # Read file into memory (not disk)
    try:
        raw_bytes = uploaded.read()

        # ROBUST ENCODING DETECTION
        detected_enc, decoded_text = detect_encoding_robust(raw_bytes)

        # Show detected encoding to user
        st.info(f"🔤 Обнаружена кодировка файла: **{detected_enc}**")

        # Parse CSV from decoded text
        from io import StringIO
        df = pd.read_csv(StringIO(decoded_text), dtype=str)

        # Clean column names (handle spaces)
        df.columns = df.columns.str.strip().str.lower()

        # Security validation
        is_safe, msg = validate_csv_security(df)
        if not is_safe:
            st.error(f"⚠️ Обнаружена потенциальная угроза безопасности: {msg}")
            st.stop()

        # Check required columns
        required = ['date', 'time', 'category', 'product', 'quantity', 'price', 'total']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(missing)}")
            st.info(f"Найденные столбцы: {', '.join(df.columns.tolist())}")
            st.stop()

        # Convert types
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['total'] = pd.to_numeric(df['total'], errors='coerce')

        # Clean strings
        for col in ['date', 'time', 'category', 'product']:
            df[col] = df[col].astype(str).str.strip()

        # Handle empty time
        df['time'] = df['time'].replace(['', 'nan', 'None', 'null'], '')
        time_empty = df['time'].eq('').all()

        # Create datetime
        datetime_str = df['date'] + (' ' + df['time'].fillna('').replace('', '00:00:00') if not time_empty else '')
        df['datetime'] = pd.to_datetime(datetime_str, errors='coerce')

        # Clean data
        df.dropna(subset=['datetime', 'quantity', 'price', 'total'], inplace=True)
        df = df[df['total'].abs() > 0]  # Allow returns (negative), but not zero
        df = df.drop_duplicates(subset=['datetime', 'category', 'product', 'quantity', 'price'])
        df.sort_values('datetime', inplace=True)

        if df.empty:
            st.error("❌ Нет данных после очистки. Проверьте формат дат и числовых значений.")
            st.stop()

        # Store in session
        session_mgr.store('df', df)

        st.success(f"✅ Успешно загружено {len(df):,} записей за период {df['datetime'].min().strftime('%d-%m-%Y')} — {df['datetime'].max().strftime('%d-%m-%Y')}")

        # Preview
        with st.expander("🔍 Предпросмотр данных (первые 10 строк)", expanded=True):
            preview_df = df.head(10).copy()
            preview_df['datetime'] = preview_df['datetime'].dt.strftime('%d-%m-%Y %H:%M')
            st.dataframe(preview_df, use_container_width=True)

        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего записей", f"{len(df):,}")
        col2.metric("Категорий", df['category'].nunique())
        col3.metric("Товаров", df['product'].nunique())
        col4.metric("Период (дней)", (df['datetime'].max() - df['datetime'].min()).days)

    except Exception as e:
        st.error(f"❌ Ошибка обработки файла: {str(e)}")
        st.info("""
        Рекомендации:
        1. Убедитесь, что файл действительно в формате CSV (разделитель — запятая)
        2. Проверьте, что первая строка содержит заголовки столбцов
        3. Попробуйте сохранить файл через Excel с кодировкой UTF-8
        4. Если файл создан в 1С или другой российской системе — попробуйте кодировку Windows-1251
        """)
        st.stop()

# Forecast configuration
if session_mgr.get('df') is not None:
    df = session_mgr.get('df')

    st.markdown("---")
    st.subheader("🔧 Параметры прогнозирования")

    col1, col2, col3, col4 = st.columns(4)

    # Frequency selection
    freq_map = {
        'Час': 'H',
        'День': 'D',
        'Неделя (с понедельника)': 'W-MON',
        'Месяц': 'MS'
    }
    freq_label = col1.selectbox(
        "Периодичность агрегации",
        list(freq_map.keys()),
        index=2,
        help="Выберите период, по которому будут агрегироваться данные"
    )
    freq = freq_map[freq_label]

    # Category filter
    cats = ['Все категории'] + sorted(df['category'].unique().tolist())
    selected_cat = col2.selectbox("Категория", cats)

    # Product filter
    if selected_cat != 'Все категории':
        prods = ['Все товары'] + sorted(df[df['category'] == selected_cat]['product'].unique().tolist())
    else:
        prods = ['Все товары']
    selected_prod = col3.selectbox("Товар", prods)

    # Horizon
    max_horizon = {'H': 168, 'D': 90, 'W-MON': 52, 'MS': 24}[freq]
    horizon = col4.number_input(
        "Горизонт прогноза",
        min_value=1,
        max_value=max_horizon,
        value=min(12, max_horizon),
        help=f"Количество периодов для прогноза (макс. {max_horizon})"
    )

    # Advanced options
    with st.expander("⚙️ Дополнительные настройки"):
        col_a1, col_a2 = st.columns(2)
        show_advanced = col_a1.checkbox("📊 Расширенная аналитика", value=False)
        show_returns = col_a2.checkbox("📉 Учитывать возвраты (отрицательные суммы)", value=False)

        if show_returns:
            df_work = df.copy()
        else:
            df_work = df[df['total'] > 0].copy()

    # Filter data
    df_f = df_work.copy()
    if selected_cat != 'Все категории':
        df_f = df_f[df_f['category'] == selected_cat]
    if selected_prod != 'Все товары':
        df_f = df_f[df_f['product'] == selected_prod]

    if df_f.empty:
        st.warning("⚠️ Нет данных для выбранной комбинации фильтров.")
        st.stop()

    # Check data sufficiency
    min_required = {'H': 48, 'D': 14, 'W-MON': 10, 'MS': 6}[freq]
    ts_check = df_f.set_index('datetime')['total'].resample(freq).sum().dropna()
    if len(ts_check) < min_required:
        st.warning(f"⚠️ Недостаточно данных для прогноза. Требуется минимум {min_required} периодов, доступно {len(ts_check)}.")
        st.stop()

    # Run forecast button
    st.markdown("---")
    if st.button("🚀 Построить прогноз", type="primary", use_container_width=True):
        start_time = time.time()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Forecast total sales
            status_text.text("📊 Анализ суммы продаж и обучение моделей...")
            progress_bar.progress(10)

            res_total = process_target(df_f, 'total', freq, horizon)

            if res_total is None:
                st.error("❌ Не удалось построить прогноз для суммы продаж. Возможно, недостаточно данных.")
                st.stop()

            progress_bar.progress(50)
            status_text.text("📦 Анализ количества продаж...")

            # Forecast quantity
            res_qty = process_target(df_f, 'quantity', freq, horizon)

            progress_bar.progress(80)
            status_text.text("📈 Формирование графиков и отчета...")

            # Store results
            session_mgr.store('res_total', res_total)
            session_mgr.store('res_qty', res_qty)

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            # =================================================================
            # RESULTS DISPLAY
            # =================================================================

            st.markdown("---")
            st.subheader(f"🏆 Результаты прогнозирования: {res_total['best_name']}")

            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("RMSE", f"{res_total['rmse']:,.2f}", help="Среднеквадратичная ошибка")
            col_m2.metric("MAPE", f"{res_total['mape']:.2f}%", help="Средняя абсолютная процентная ошибка")
            col_m3.metric("SMAPE", f"{res_total['smape']:.2f}%", help="Симметричная MAPE")
            col_m4.metric("MAE", f"{res_total['mae']:,.2f}", help="Средняя абсолютная ошибка")

            # Alternative models comparison
            other_models = [m for m in res_total['models'] if m != res_total['best_name']]
            if other_models:
                best_alt = min(other_models, key=lambda x: res_total['models'][x]['mape'] if res_total['models'][x]['mape'] != np.inf else 999999)
                st.caption(f"📌 Альтернатива: **{best_alt}** (MAPE: {res_total['models'][best_alt]['mape']:.2f}%)")

            # Main chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=res_total['train'].index,
                y=res_total['train'].values,
                name='Исторические данные (обучение)',
                line=dict(color='#1f77b4', width=2),
                mode='lines'
            ))

            fig.add_trace(go.Scatter(
                x=res_total['test'].index,
                y=res_total['test'].values,
                name='Исторические данные (тест)',
                line=dict(color='#ff7f0e', width=2),
                mode='lines'
            ))

            fig.add_trace(go.Scatter(
                x=res_total['future'],
                y=res_total['forecast'],
                name='Прогноз',
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers'
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([res_total['future'], res_total['future'][::-1]]),
                y=np.concatenate([res_total['upper'], res_total['lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% доверительный интервал',
                hoverinfo='skip'
            ))

            # Split line
            split_date = res_total['test'].index[0] if len(res_total['test']) > 0 else res_total['train'].index[-1]
            fig.add_vline(
                x=split_date,
                line=dict(color='red', dash='dash', width=2),
                annotation_text="Начало прогноза",
                annotation_position="top"
            )

            fig.update_layout(
                title=f'Прогноз суммы продаж — {res_total["best_name"]}',
                xaxis_title='Дата',
                yaxis_title='Сумма продаж',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # Forecast table
            st.subheader("📋 Прогнозные значения")

            table_data = {
                'Дата': res_total['future'].strftime('%d-%m-%Y'),
                'Прогноз суммы': res_total['forecast'].round(2),
                'Нижняя граница (90%)': res_total['lower'].round(2),
                'Верхняя граница (90%)': res_total['upper'].round(2)
            }

            if res_qty is not None:
                table_data['Прогноз количества'] = res_qty['forecast'].round(0).astype(int)
                table_data['Нижняя граница кол-ва'] = np.maximum(res_qty['lower'], 0).round(0).astype(int)
                table_data['Верхняя граница кол-ва'] = res_qty['upper'].round(0).astype(int)

            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True)

            # Download CSV
            csv_buffer = io.StringIO()
            table_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать прогноз в CSV",
                data=csv_buffer.getvalue(),
                file_name=f"forecast_{session_mgr.session_id}.csv",
                mime="text/csv"
            )

            # PDF Report
            if HAS_REPORTLAB:
                pdf_data = generate_pdf_report(res_total, res_qty, df_f, horizon, freq_label)
                if pdf_data:
                    st.download_button(
                        label="📄 Скачать отчет в PDF",
                        data=pdf_data,
                        file_name=f"report_{session_mgr.session_id}.pdf",
                        mime="application/pdf"
                    )

            # =================================================================
            # ADVANCED ANALYTICS
            # =================================================================

            if show_advanced:
                st.markdown("---")
                st.subheader("📊 Расширенная аналитика")

                # Model comparison
                with st.expander("📈 Сравнение моделей"):
                    comp_data = []
                    for name, data in res_total['models'].items():
                        comp_data.append({
                            'Модель': name,
                            'RMSE': data['rmse'],
                            'MAPE (%)': data['mape'],
                            'SMAPE (%)': data['smape'],
                            'MAE': data['mae']
                        })

                    comp_df = pd.DataFrame(comp_data).sort_values('MAPE (%)')
                    st.dataframe(comp_df, use_container_width=True)

                    # Comparison chart
                    fig_comp = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('RMSE', 'MAPE (%)', 'SMAPE (%)', 'MAE'),
                        vertical_spacing=0.15
                    )

                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['RMSE'], name='RMSE'), 1, 1)
                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['MAPE (%)'], name='MAPE'), 1, 2)
                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['SMAPE (%)'], name='SMAPE'), 2, 1)
                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['MAE'], name='MAE'), 2, 2)

                    fig_comp.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_comp, use_container_width=True)

                # Seasonal decomposition
                if len(res_total['train']) >= 2 * res_total['sp'] + 5:
                    with st.expander("🔄 Сезонная декомпозиция"):
                        try:
                            dec = seasonal_decompose(
                                res_total['train'],
                                model='additive',
                                period=res_total['sp']
                            )

                            fig_dec = make_subplots(
                                rows=4, cols=1,
                                subplot_titles=('Наблюдения', 'Тренд', 'Сезонность', 'Остатки'),
                                vertical_spacing=0.08
                            )

                            fig_dec.add_trace(go.Scatter(x=res_total['train'].index, y=dec.observed, name='Наблюдения'), 1, 1)
                            fig_dec.add_trace(go.Scatter(x=res_total['train'].index, y=dec.trend, name='Тренд'), 2, 1)
                            fig_dec.add_trace(go.Scatter(x=res_total['train'].index, y=dec.seasonal, name='Сезонность'), 3, 1)
                            fig_dec.add_trace(go.Scatter(x=res_total['train'].index, y=dec.resid, name='Остатки'), 4, 1)

                            fig_dec.update_layout(height=900, showlegend=False)
                            st.plotly_chart(fig_dec, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Не удалось выполнить декомпозицию: {e}")

                # Residuals analysis
                with st.expander("📉 Анализ остатков"):
                    residuals = res_total['residuals']

                    col_r1, col_r2 = st.columns(2)

                    # Residuals plot
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(
                        x=res_total['test'].index,
                        y=residuals,
                        mode='markers',
                        name='Остатки',
                        marker=dict(color='blue', size=8)
                    ))
                    fig_res.add_hline(y=0, line_dash='dash', line_color='red')
                    fig_res.update_layout(
                        title='Остатки модели (тестовая выборка)',
                        xaxis_title='Дата',
                        yaxis_title='Ошибка'
                    )
                    col_r1.plotly_chart(fig_res, use_container_width=True)

                    # Residuals histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=20,
                        name='Распределение остатков'
                    ))
                    fig_hist.update_layout(
                        title='Распределение остатков',
                        xaxis_title='Ошибка',
                        yaxis_title='Частота'
                    )
                    col_r2.plotly_chart(fig_hist, use_container_width=True)

                    # ACF
                    if len(residuals) > 5:
                        acf_vals = acf(residuals, nlags=min(20, len(residuals)//2))

                        fig_acf = go.Figure()
                        for i, val in enumerate(acf_vals):
                            fig_acf.add_vline(x=i, line_width=3, line_color='blue', opacity=abs(val))

                        conf_level = 1.96 / np.sqrt(len(residuals))
                        fig_acf.add_hline(y=conf_level, line_dash='dash', line_color='red', annotation_text="95% граница")
                        fig_acf.add_hline(y=-conf_level, line_dash='dash', line_color='red')
                        fig_acf.add_hline(y=0, line_color='black')

                        fig_acf.update_layout(
                            title='Автокорреляция остатков (ACF)',
                            xaxis_title='Лаг',
                            yaxis_title='ACF'
                        )
                        st.plotly_chart(fig_acf, use_container_width=True)

                        st.caption("""
                        Значимые пики за пределами красных линий указывают на оставшуюся 
                        структуру в ошибках. Идеально — остатки похожи на белый шум.
                        """)

                # Feature importance
                if res_total['best_name'] not in ('Holt-Winters', 'Naive', 'Seasonal Naive'):
                    if res_total['X_train_for_best'] is not None and hasattr(
                        res_total['models'][res_total['best_name']]['model'], 'feature_importances_'
                    ):
                        with st.expander("🔍 Важность признаков"):
                            model_obj = res_total['models'][res_total['best_name']]['model']
                            importances = model_obj.feature_importances_
                            feat_names = res_total['X_train_for_best'].columns

                            imp_df = pd.DataFrame({
                                'Признак': feat_names,
                                'Важность': importances
                            }).sort_values('Важность', ascending=True)

                            fig_imp = go.Figure(go.Bar(
                                x=imp_df['Важность'],
                                y=imp_df['Признак'],
                                orientation='h',
                                marker_color='#1f77b4'
                            ))
                            fig_imp.update_layout(
                                title='Важность признаков в лучшей модели',
                                xaxis_title='Важность',
                                height=400
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)

                # Correlation matrix
                if res_total['X_train_for_best'] is not None:
                    with st.expander("🔗 Корреляционная матрица"):
                        X_corr = res_total['X_train_for_best'].copy()
                        y_corr = res_total['train'].loc[X_corr.index]
                        X_corr['target'] = y_corr

                        corr = X_corr.corr()

                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.index,
                            colorscale='RdBu_r',
                            zmin=-1,
                            zmax=1,
                            text=np.round(corr.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10}
                        ))
                        fig_corr.update_layout(
                            title='Корреляции признаков и целевой переменной',
                            height=500
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

            # Execution time
            elapsed = time.time() - start_time
            st.caption(f"⏱️ Прогноз построен за {elapsed:.1f} секунд | Сессия: {session_mgr.session_id}")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Ошибка при построении прогноза: {str(e)}")
            st.info("""
            Рекомендации по устранению:
            1. Проверьте достаточность данных (минимум 10 периодов)
            2. Попробуйте другую периодичность агрегации
            3. Уменьшите горизонт прогноза
            4. Проверьте наличие выбросов в данных
            """)

        finally:
            # Cleanup
            if 'df_f' in locals():
                del df_f
            gc.collect()

# Footer
st.markdown("---")
st.caption("""
🔒 **Безопасность:** Данные обрабатываются только в оперативной памяти и удаляются после сессии. 
Никакое логирование или сохранение на диск не производится.
""")
