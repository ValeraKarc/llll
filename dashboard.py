import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecast System", layout="wide")

st.title("📊 Интеллектуальная система прогнозирования продаж")

# =========================
# 1. ЗАГРУЗКА CSV (КЕШ)
# =========================
@st.cache_data(show_spinner=False)
def load_csv(file):

    encodings = ["utf-8", "cp1251", "latin1"]

    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, low_memory=False)
        except:
            continue

    raise ValueError("Ошибка чтения файла")


# =========================
# 2. АВТО-ОПРЕДЕЛЕНИЕ КОЛОНОК
# =========================
def detect_columns(df):

    col_map = {}
    columns = {col.lower(): col for col in df.columns}

    patterns = {
        "date": ["date", "day"],
        "time": ["time", "hour"],
        "category": ["category", "type"],
        "product": ["product", "item", "name"],
        "quantity": ["quantity", "qty", "count"],
        "price": ["price", "cost"],
        "total": ["total", "sum", "revenue", "amount"]
    }

    for key, variants in patterns.items():
        for v in variants:
            for col_lower, original in columns.items():
                if v in col_lower:
                    col_map[key] = original
                    break
            if key in col_map:
                break

    return col_map


# =========================
# 3. UI
# =========================
file = st.file_uploader("📂 Загрузите CSV файл", type="csv")

if file:

    df = load_csv(file)

    st.success("Файл загружен")
    st.subheader("📌 Превью")
    st.dataframe(df.head(10))

    # =========================
    # АВТО-МАППИНГ
    # =========================
    col_map = detect_columns(df)

    st.subheader("🔍 Определённые колонки")
    st.write(col_map)

    # =========================
    # РУЧНАЯ КОРРЕКЦИЯ
    # =========================
    st.subheader("⚙️ Проверьте и исправьте (если нужно)")

    def select_col(name):
        return st.selectbox(name, ["—"] + list(df.columns),
                            index=list(df.columns).index(col_map[name]) if name in col_map else 0)

    date_col = select_col("date")
    time_col = select_col("time")
    total_col = select_col("total")

    # optional
    category_col = st.selectbox("category", ["—"] + list(df.columns))
    product_col = st.selectbox("product", ["—"] + list(df.columns))

    # =========================
    # ВАЛИДАЦИЯ
    # =========================
    if "—" in [date_col, time_col, total_col]:
        st.warning("Выберите обязательные поля")
        st.stop()

    # =========================
    # ОБРАБОТКА
    # =========================
    df["datetime"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    # фильтр категории
    if category_col != "—":
        categories = ["All"] + list(df[category_col].unique())
        selected_cat = st.selectbox("Фильтр категории", categories)

        if selected_cat != "All":
            df = df[df[category_col] == selected_cat]

    # =========================
    # АГРЕГАЦИЯ
    # =========================
    freq = st.selectbox("Период агрегации", ["D", "W", "M"])

    ts = (
        df.set_index("datetime")[total_col]
        .resample(freq)
        .sum()
        .fillna(0)
    )

    # =========================
    # ВИЗУАЛИЗАЦИЯ
    # =========================
    st.subheader("📈 Продажи")
    st.line_chart(ts)

    # =========================
    # МЕТРИКИ
    # =========================
    st.subheader("📊 Метрики")

    col1, col2, col3 = st.columns(3)

    col1.metric("Среднее", f"{ts.mean():.2f}")
    col2.metric("Максимум", f"{ts.max():.2f}")
    col3.metric("Минимум", f"{ts.min():.2f}")
