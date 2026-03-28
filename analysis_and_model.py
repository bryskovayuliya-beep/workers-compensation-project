import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.title("Прогнозирование стоимости страховых выплат")
st.write("Нажмите кнопку ниже, чтобы загрузить данные, обучить модель и посмотреть результаты.")

if st.button("Загрузить данные и обучить модель"):
    with st.spinner("Загрузка данных..."):
        data = fetch_openml(data_id=42876, as_frame=True, parser="auto")
        df = data.frame.copy()

    st.success("Данные успешно загружены")

    st.subheader("Первые строки датасета")
    st.write(df.head())

    st.subheader("Размер данных")
    st.write(df.shape)

    # Преобразование дат
    df["DateTimeOfAccident"] = pd.to_datetime(df["DateTimeOfAccident"])
    df["DateReported"] = pd.to_datetime(df["DateReported"])

    # Создание новых признаков из дат
    df["AccidentMonth"] = df["DateTimeOfAccident"].dt.month
    df["AccidentDayOfWeek"] = df["DateTimeOfAccident"].dt.dayofweek
    df["ReportingDelay"] = (df["DateReported"] - df["DateTimeOfAccident"]).dt.days

    # Удаление исходных datetime-столбцов
    df = df.drop(columns=["DateTimeOfAccident", "DateReported"])

    # Кодирование категориальных признаков
    categorical_columns = [
        "Gender",
        "MaritalStatus",
        "PartTimeFullTime",
        "ClaimDescription"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Признаки и целевая переменная
    X = df.drop(columns=["UltimateIncurredClaimCost"])
    y = df["UltimateIncurredClaimCost"]

    # Уменьшаем размер выборки для более быстрого обучения
    X = X.sample(n=10000, random_state=42)
    y = y.loc[X.index]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Разделение данных")
    st.write(f"Размер обучающей выборки: {X_train.shape}")
    st.write(f"Размер тестовой выборки: {X_test.shape}")

    # Обучение модели
    st.write("Началось обучение модели...")
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    st.success("Модель обучена")

    # Предсказания
    y_pred = model.predict(X_test)

    # Метрики
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Метрики модели")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.4f}")

    # График: реальные и предсказанные значения
    st.subheader("Сравнение реальных и предсказанных значений")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )
    ax.set_xlabel("Реальные значения")
    ax.set_ylabel("Предсказанные значения")
    ax.set_title("Предсказания vs Реальные значения")
    st.pyplot(fig)

    # Важность признаков
    st.subheader("Топ-10 важных признаков")
    importance_df = pd.DataFrame({
        "Признак": X_train.columns,
        "Важность": model.feature_importances_
    }).sort_values("Важность", ascending=False)

    st.write(importance_df.head(10))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    top_features = importance_df.head(10)
    ax2.barh(top_features["Признак"], top_features["Важность"])
    ax2.invert_yaxis()
    ax2.set_title("Важность признаков")
    st.pyplot(fig2)