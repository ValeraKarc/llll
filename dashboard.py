# Установка дополнительных библиотек (xgboost, pmdarima, holidays, reportlab, plotly)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from ipywidgets import widgets, interact, fixed, Layout, VBox, HBox, Output
from IPython.display import display, clear_output, FileLink, HTML
import holidays
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# Сброс всех переменных перед запуском (для чистоты)
%reset -f

print("✅ Библиотеки загружены. Загрузите CSV-файл.")

# Основная логика прогнозирования с интерфейсом
from google.colab import files
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from ipywidgets import *
import warnings
warnings.filterwarnings('ignore')

# Глобальные переменные для хранения данных
uploaded_df = None
forecast_results = {}

def validate_data(df):
    required = ['date', 'time', 'category', 'product', 'quantity', 'price', 'total']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
    if df['datetime'].isnull().any():
        raise ValueError("Неверный формат даты/времени. Ожидается ДД-ММ-ГГГГ ЧЧ:ММ:СС")
    for col in ['quantity','price','total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['quantity','price','total']].isnull().any().any():
        raise ValueError("Колонки quantity, price, total должны быть числами")
    df['total'] = df['quantity'] * df['price']  # пересчёт для согласованности
    df.sort_values('datetime', inplace=True)
    return df

def aggregate_series(df, freq, category=None, product=None):
    data = df.copy()
    if category and category != 'Все':
        data = data[data['category'] == category]
    if product and product != 'Все':
        data = data[data['product'] == product]
    if data.empty:
        raise ValueError("Нет данных после фильтрации")
    series = data.set_index('datetime').resample(freq)['total'].sum().fillna(0)
    return series

def create_features(ts, freq, is_holiday_func):
    df = pd.DataFrame({'y': ts})
    for lag in [1,2,3,7]:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_mean_7'] = df['y'].rolling(7, min_periods=1).mean()
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['hour'] = df.index.hour if freq == 'H' else 0
    df['is_holiday'] = df.index.map(lambda d: 1 if is_holiday_func(d) else 0)
    df.dropna(inplace=True)
    return df

def forecast_ml(model_class, params, train, horizon, freq, is_holiday_func):
    # рекурсивный прогноз для ML-моделей
    model = model_class(**params)
    feats = create_features(train, freq, is_holiday_func)
    X = feats.drop('y', axis=1)
    y = feats['y']
    model.fit(X, y)
    pred = []
    last = train.copy()
    for step in range(horizon):
        next_date = last.index[-1] + pd.Timedelta(freq)
        new = pd.DataFrame(index=[next_date])
        for lag in [1,2,3,7]:
            val = last.iloc[-lag] if len(last) >= lag else 0
            new[f'lag_{lag}'] = val
        new['rolling_mean_7'] = last[-7:].mean() if len(last)>=7 else last.mean()
        new['dayofweek'] = next_date.dayofweek
        new['month'] = next_date.month
        new['hour'] = next_date.hour
        new['is_holiday'] = 1 if is_holiday_func(next_date) else 0
        p = model.predict(new)[0]
        pred.append(p)
        last = pd.concat([last, pd.Series([p], index=[next_date])])
    return pred, model

def run_forecast(series, horizon, freq, is_holiday_func):
    if len(series) < 2*horizon and len(series) > horizon:
        train = series
        test = None
    elif len(series) <= horizon:
        raise ValueError("Недостаточно данных для выбранного горизонта")
    else:
        train = series.iloc[:-horizon]
        test = series.iloc[-horizon:]

    models = {}
    # ETS
    try:
        seasonal = {'H':24, 'D':7, 'W':52, 'M':12}.get(freq, None)
        if seasonal:
            model_ets = ExponentialSmoothing(train, seasonal_periods=seasonal, trend='add', seasonal='add')
            fit = model_ets.fit()
            pred = fit.forecast(horizon)
            mape = mean_absolute_percentage_error(test, pred[:len(test)]) if test is not None else np.nan
            rmse = np.sqrt(mean_squared_error(test, pred[:len(test)])) if test is not None else np.nan
            models['ETS'] = {'model': fit, 'forecast': pred, 'mape': mape, 'rmse': rmse}
    except: pass

    # ARIMA
    try:
        auto = pm.auto_arima(train, seasonal=True, m=7 if freq=='D' else 1, trace=False, error_action='ignore', suppress_warnings=True)
        pred = auto.predict(horizon)
        mape = mean_absolute_percentage_error(test, pred[:len(test)]) if test is not None else np.nan
        rmse = np.sqrt(mean_squared_error(test, pred[:len(test)])) if test is not None else np.nan
        models['ARIMA'] = {'model': auto, 'forecast': pred, 'mape': mape, 'rmse': rmse}
    except: pass

    # RandomForest
    try:
        pred, model = forecast_ml(RandomForestRegressor, {'n_estimators':100, 'random_state':42}, train, horizon, freq, is_holiday_func)
        mape = mean_absolute_percentage_error(test, pred[:len(test)]) if test is not None else np.nan
        rmse = np.sqrt(mean_squared_error(test, pred[:len(test)])) if test is not None else np.nan
        models['RandomForest'] = {'model': model, 'forecast': pred, 'mape': mape, 'rmse': rmse}
    except: pass

    # XGBoost
    try:
        pred, model = forecast_ml(XGBRegressor, {'n_estimators':100, 'random_state':42, 'verbosity':0}, train, horizon, freq, is_holiday_func)
        mape = mean_absolute_percentage_error(test, pred[:len(test)]) if test is not None else np.nan
        rmse = np.sqrt(mean_squared_error(test, pred[:len(test)])) if test is not None else np.nan
        models['XGBoost'] = {'model': model, 'forecast': pred, 'mape': mape, 'rmse': rmse}
    except: pass

    if not models:
        raise ValueError("Не удалось обучить ни одну модель")
    # выбор лучшей по MAPE или по RMSE, если MAPE нет
    best = min(models.items(), key=lambda x: x[1]['mape'] if not np.isnan(x[1]['mape']) else x[1]['rmse'])
    best_name, best_res = best
    return best_name, best_res, models, train, test

def get_confidence_interval(best_name, best_res, train, freq, is_holiday_func):
    if best_name in ['ETS', 'ARIMA']:
        if best_name == 'ETS':
            fitted = best_res['model'].fittedvalues
        else:
            fitted = best_res['model'].predict_in_sample()
        residuals = train.values[-len(fitted):] - fitted
    else:
        feats = create_features(train, freq, is_holiday_func)
        X = feats.drop('y', axis=1)
        pred_train = best_res['model'].predict(X)
        residuals = feats['y'].values - pred_train
    sigma = np.std(residuals)
    z = 1.645
    lower = [p - z*sigma for p in best_res['forecast']]
    upper = [p + z*sigma for p in best_res['forecast']]
    return lower, upper

# --- Интерфейс ---
out = Output()
upload_widget = widgets.FileUpload(accept='.csv', multiple=False, description='Выберите CSV')

freq_map = {'Час': 'H', 'День': 'D', 'Неделя': 'W', 'Месяц': 'M'}
freq_drop = widgets.Dropdown(options=list(freq_map.keys()), value='День', description='Агрегация')
cat_drop = widgets.Dropdown(options=['Все'], description='Категория')
prod_drop = widgets.Dropdown(options=['Все'], description='Товар')
horizon_slider = widgets.IntSlider(value=30, min=1, max=365, description='Горизонт')
run_btn = widgets.Button(description='🚀 Создать прогноз', button_style='primary')
status = widgets.HTML(value='')

def update_filters(df):
    if df is not None:
        cats = ['Все'] + sorted(df['category'].unique())
        prods = ['Все'] + sorted(df['product'].unique())
        cat_drop.options = cats
        prod_drop.options = prods

def on_upload_change(change):
    global uploaded_df
    if upload_widget.value:
        content = list(upload_widget.value.values())[0]['content']
        try:
            df = pd.read_csv(io.BytesIO(content))
            df = validate_data(df)
            uploaded_df = df
            update_filters(df)
            with out:
                clear_output()
                display(HTML(f"✅ Загружено {len(df)} строк.<br>Предпросмотр:"))
                display(df.head())
        except Exception as e:
            with out:
                clear_output()
                display(HTML(f"❌ Ошибка: {str(e)}"))

upload_widget.observe(on_upload_change, names='value')

def on_run_click(b):
    global uploaded_df, forecast_results
    if uploaded_df is None:
        status.value = "⚠️ Сначала загрузите CSV файл"
        return
    freq = freq_map[freq_drop.value]
    cat = cat_drop.value
    prod = prod_drop.value
    horizon = horizon_slider.value
    status.value = "⏳ Выполняется прогноз, подождите..."
    with out:
        clear_output(wait=True)
        try:
            series = aggregate_series(uploaded_df, freq, cat, prod)
            holidays_ru = holidays.Russia()
            def is_holiday_func(dt): return dt.date() in holidays_ru
            best_name, best_res, all_models, train, test = run_forecast(series, horizon, freq, is_holiday_func)
            lower, upper = get_confidence_interval(best_name, best_res, train, freq, is_holiday_func)
            forecast_results = {
                'series': series,
                'best_name': best_name,
                'forecast': best_res['forecast'],
                'lower': lower,
                'upper': upper,
                'mape': best_res['mape'],
                'rmse': best_res['rmse'],
                'horizon': horizon,
                'freq': freq,
                'cat': cat,
                'prod': prod,
                'all_models': all_models,
                'forecast_values': best_res['forecast']
            }
            # График
            last_date = series.index[-1]
            delta = pd.Timedelta(freq)
            forecast_dates = [last_date + (i+1)*delta for i in range(horizon)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name='История'))
            fig.add_trace(go.Scatter(x=forecast_dates, y=best_res['forecast'], name='Прогноз', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=forecast_dates, y=upper, name='Верхняя граница (90%)', line=dict(dash='dash', color='gray'), showlegend=True))
            fig.add_trace(go.Scatter(x=forecast_dates, y=lower, name='Нижняя граница (90%)', line=dict(dash='dash', color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
            fig.update_layout(title=f'Прогноз продаж (лучшая модель: {best_name})', xaxis_title='Дата', yaxis_title='Сумма продаж', height=500)
            pio.show(fig)

            # Таблица прогноза
            forecast_table = pd.DataFrame({
                'Дата': [d.strftime('%d-%m-%Y %H:%M' if freq=='H' else '%d-%m-%Y') for d in forecast_dates],
                'Прогноз': np.round(best_res['forecast'], 2),
                'Нижняя граница (90%)': np.round(lower, 2),
                'Верхняя граница (90%)': np.round(upper, 2)
            })
            display(HTML(f"<h3>Метрики</h3><b>Лучшая модель:</b> {best_name}<br>"
                         f"<b>MAPE:</b> {best_res['mape']:.2%} &nbsp;&nbsp; <b>RMSE:</b> {best_res['rmse']:.2f}"))
            display(HTML("<h3>Таблица прогноза (первые 20 строк)</h3>"))
            display(forecast_table.head(20))

            # Кнопка скачивания CSV
            csv_buffer = io.BytesIO()
            forecast_table.to_csv(csv_buffer, index=False)
            csv_b64 = base64.b64encode(csv_buffer.getvalue()).decode()
            href = f'<a href="data:text/csv;base64,{csv_b64}" download="forecast.csv">📥 Скачать прогноз (CSV)</a>'
            display(HTML(href))

            # PDF отчёт (упрощённый через matplotlib сохраним в PNG и вставим)
            # Создаём временный PDF
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            story.append(Paragraph("Отчёт по прогнозированию продаж", styles['Title']))
            story.append(Spacer(1,12))
            story.append(Paragraph(f"Агрегация: {freq_drop.value}, Категория: {cat}, Товар: {prod}", styles['Normal']))
            story.append(Paragraph(f"Горизонт: {horizon}, Лучшая модель: {best_name}, MAPE: {best_res['mape']:.2%}, RMSE: {best_res['rmse']:.2f}", styles['Normal']))
            story.append(Spacer(1,12))
            story.append(Paragraph("Таблица прогноза", styles['Heading2']))
            table_data = [list(forecast_table.columns)]
            for row in forecast_table.values[:20]:
                table_data.append([str(x) for x in row])
            t = Table(table_data)
            t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke), ('ALIGN',(0,0),(-1,-1),'CENTER'), ('GRID',(0,0),(-1,-1),1,colors.black)]))
            story.append(t)
            story.append(Spacer(1,12))
            # Сохраняем график Plotly во временный PNG
            img_bytes = fig.to_image(format="png", width=800, height=400)
            img_io = io.BytesIO(img_bytes)
            story.append(Paragraph("График прогноза", styles['Heading2']))
            story.append(Image(img_io, width=500, height=250))
            doc.build(story)
            pdf_buffer.seek(0)
            pdf_b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            pdf_href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="forecast_report.pdf">📄 Скачать PDF-отчёт</a>'
            display(HTML(pdf_href))

            status.value = "✅ Прогноз выполнен успешно"
        except Exception as e:
            status.value = f"❌ Ошибка: {str(e)}"
            display(HTML(f"<span style='color:red'>{str(e)}</span>"))

run_btn.on_click(on_run_click)

# Отображение интерфейса
display(HTML("<h2>📊 Система прогнозирования продаж</h2>"))
display(upload_widget)
display(HBox([freq_drop, horizon_slider]))
display(HBox([cat_drop, prod_drop]))
display(run_btn)
display(status)
display(out)
