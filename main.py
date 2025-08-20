"""
Aplicación de Machine Learning con Streamlit
===========================================

Esta aplicación demuestra un modelo supervisado de clasificación para predecir
la compra de productos basado en características del cliente.

Autor: T3 Chat Assistant
Fecha: 20/08/2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras - ML Demo",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def generar_datos_simulados(n_samples=500):
    """
    Genera un dataset simulado para predicción de compras de clientes.

    Args:
        n_samples (int): Número de muestras a generar (mínimo 300)

    Returns:
        pd.DataFrame: Dataset con características del cliente y target
    """
    np.random.seed(42)

    # Generar características del cliente
    edad = np.random.normal(35, 12, n_samples).clip(18, 80)
    ingresos = np.random.lognormal(10, 0.5, n_samples).clip(20000, 200000)
    tiempo_web = np.random.exponential(15, n_samples).clip(1, 120)
    num_productos_vistos = np.random.poisson(8, n_samples).clip(1, 30)
    historial_compras = np.random.poisson(3, n_samples).clip(0, 20)
    puntuacion_credito = np.random.normal(650, 100, n_samples).clip(300, 850)

    # Crear variable target basada en lógica de negocio
    # La probabilidad de compra aumenta con:
    # - Mayor edad (hasta cierto punto)
    # - Mayores ingresos
    # - Más tiempo en web
    # - Más productos vistos
    # - Más historial de compras
    # - Mejor puntuación de crédito

    prob_compra = (
        0.1 * (edad - 18) / 62
        + 0.3 * np.log(ingresos / 20000) / np.log(10)
        + 0.2 * np.minimum(tiempo_web / 60, 1)
        + 0.2 * np.minimum(num_productos_vistos / 15, 1)
        + 0.15 * np.minimum(historial_compras / 10, 1)
        + 0.05 * (puntuacion_credito - 300) / 550
    )

    # Añadir ruido aleatorio
    prob_compra += np.random.normal(0, 0.1, n_samples)
    prob_compra = np.clip(prob_compra, 0, 1)

    # Generar variable target
    compra = np.random.binomial(1, prob_compra, n_samples)

    # Crear DataFrame
    data = pd.DataFrame(
        {
            "edad": edad.round(0).astype(int),
            "ingresos_anuales": ingresos.round(0).astype(int),
            "tiempo_web_minutos": tiempo_web.round(1),
            "productos_vistos": num_productos_vistos,
            "historial_compras": historial_compras,
            "puntuacion_credito": puntuacion_credito.round(0).astype(int),
            "compra": compra,
        }
    )

    return data


@st.cache_data
def entrenar_modelos(data):
    """
    Entrena múltiples modelos de ML y devuelve métricas.

    Args:
        data (pd.DataFrame): Dataset para entrenar

    Returns:
        dict: Diccionario con modelos entrenados y métricas
    """
    # Separar características y target
    X = data.drop("compra", axis=1)
    y = data["compra"]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar modelos
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        ),
        "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        if nombre == "Regresión Logística":
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]

        # Calcular métricas
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        resultados[nombre] = {
            "modelo": modelo,
            "metricas": metricas,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "scaler": scaler if nombre == "Regresión Logística" else None,
        }

    return resultados, X_train, X_test


def crear_graficos_eda(data):
    """
    Crea gráficos para análisis exploratorio de datos.

    Args:
        data (pd.DataFrame): Dataset a analizar
    """
    col1, col2 = st.columns(2)

    with col1:
        # Distribución de la variable target
        fig_target = px.histogram(
            data,
            x="compra",
            title="Distribución de Compras",
            labels={"compra": "Compra (0=No, 1=Sí)", "count": "Cantidad"},
        )
        st.plotly_chart(fig_target, use_container_width=True)

        # Correlación entre variables
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Matriz de Correlación")
        st.pyplot(fig_corr)

    with col2:
        # Distribución de edad por compra
        fig_edad = px.box(
            data,
            x="compra",
            y="edad",
            title="Distribución de Edad por Compra",
            labels={"compra": "Compra (0=No, 1=Sí)", "edad": "Edad"},
        )
        st.plotly_chart(fig_edad, use_container_width=True)

        # Ingresos vs Tiempo en web
        fig_scatter = px.scatter(
            data,
            x="ingresos_anuales",
            y="tiempo_web_minutos",
            color="compra",
            title="Ingresos vs Tiempo en Web",
            labels={
                "ingresos_anuales": "Ingresos Anuales",
                "tiempo_web_minutos": "Tiempo en Web (min)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def mostrar_metricas_modelo(resultados):
    """
    Muestra métricas de rendimiento de los modelos.

    Args:
        resultados (dict): Resultados de los modelos entrenados
    """
    for nombre, resultado in resultados.items():
        st.subheader(f"📊 Métricas - {nombre}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{resultado['metricas']['accuracy']:.3f}")

        with col2:
            st.metric("Precision", f"{resultado['metricas']['precision']:.3f}")

        with col3:
            st.metric("Recall", f"{resultado['metricas']['recall']:.3f}")

        with col4:
            st.metric("F1-Score", f"{resultado['metricas']['f1']:.3f}")

        # Matriz de confusión
        fig_cm = plt.figure(figsize=(6, 5))
        sns.heatmap(
            resultado["metricas"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Compra", "Compra"],
            yticklabels=["No Compra", "Compra"],
        )
        plt.title(f"Matriz de Confusión - {nombre}")
        plt.ylabel("Valor Real")
        plt.xlabel("Predicción")
        st.pyplot(fig_cm)

        st.markdown("---")


def seccion_prediccion(resultados, data):
    """
    Crea una sección interactiva para hacer predicciones.

    Args:
        resultados (dict): Modelos entrenados
        data (pd.DataFrame): Dataset original para rangos
    """
    st.header("🔮 Hacer Predicción")

    # Seleccionar modelo
    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo:", list(resultados.keys())
    )

    col1, col2 = st.columns(2)

    with col1:
        edad = st.slider(
            "Edad:", int(data["edad"].min()), int(data["edad"].max()), 35
        )
        ingresos = st.slider(
            "Ingresos Anuales:",
            int(data["ingresos_anuales"].min()),
            int(data["ingresos_anuales"].max()),
            50000,
        )
        tiempo_web = st.slider(
            "Tiempo en Web (min):",
            float(data["tiempo_web_minutos"].min()),
            float(data["tiempo_web_minutos"].max()),
            15.0,
        )

    with col2:
        productos_vistos = st.slider(
            "Productos Vistos:",
            int(data["productos_vistos"].min()),
            int(data["productos_vistos"].max()),
            8,
        )
        historial_compras = st.slider(
            "Historial de Compras:",
            int(data["historial_compras"].min()),
            int(data["historial_compras"].max()),
            3,
        )
        puntuacion_credito = st.slider(
            "Puntuación de Crédito:",
            int(data["puntuacion_credito"].min()),
            int(data["puntuacion_credito"].max()),
            650,
        )

    if st.button("🚀 Hacer Predicción"):
        # Preparar datos para predicción
        datos_prediccion = np.array(
            [
                [
                    edad,
                    ingresos,
                    tiempo_web,
                    productos_vistos,
                    historial_compras,
                    puntuacion_credito,
                ]
            ]
        )

        modelo = resultados[modelo_seleccionado]["modelo"]
        scaler = resultados[modelo_seleccionado]["scaler"]

        # Aplicar escalado si es necesario
        if scaler is not None:
            datos_prediccion = scaler.transform(datos_prediccion)

        # Hacer predicción
        prediccion = modelo.predict(datos_prediccion)[0]
        probabilidad = modelo.predict_proba(datos_prediccion)[0]

        # Mostrar resultados
        col1, col2 = st.columns(2)

        with col1:
            if prediccion == 1:
                st.success("✅ **Predicción: COMPRARÁ**")
            else:
                st.error("❌ **Predicción: NO COMPRARÁ**")

        with col2:
            st.info(f"🎯 **Probabilidad de Compra: {probabilidad[1]:.2%}**")

        # Gráfico de probabilidades
        fig_prob = go.Figure(
            data=[
                go.Bar(
                    x=["No Compra", "Compra"],
                    y=[probabilidad[0], probabilidad[1]],
                    marker_color=["red", "green"],
                )
            ]
        )
        fig_prob.update_layout(title="Probabilidades de Predicción", showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)


def main():
    """
    Función principal de la aplicación Streamlit.
    """
    # Título principal
    st.title("🛒 Predictor de Compras - ML Demo")
    st.markdown(
        """
    Esta aplicación demuestra un **modelo supervisado de Machine Learning** 
    para predecir si un cliente realizará una compra basado en sus características.
    """
    )

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Proyecto")
        st.markdown(
            """
        **Características del Dataset:**
        - 🔢 Muestras: 500
        - 📊 Variables: 6 características + 1 target
        - 🎯 Tipo: Clasificación binaria
        - 📈 Modelos: Random Forest y Regresión Logística
        
        **Variables:**
        - Edad del cliente
        - Ingresos anuales
        - Tiempo en sitio web
        - Productos vistos
        - Historial de compras
        - Puntuación crediticia
        """
        )

        # Control del tamaño del dataset
        n_samples = st.slider("Número de muestras:", 300, 1000, 500)

    # Generar datos
    with st.spinner("🔄 Generando dataset simulado..."):
        data = generar_datos_simulados(n_samples)

    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Exploración de Datos", "🤖 Modelos ML", "📈 Métricas", "🔮 Predicción"]
    )

    with tab1:
        st.header("📊 Análisis Exploratorio de Datos")

        # Estadísticas descriptivas
        st.subheader("📋 Estadísticas Descriptivas")
        st.dataframe(data.describe())

        # Mostrar muestra del dataset
        st.subheader("👀 Muestra del Dataset")
        st.dataframe(data.head(10))

        # Gráficos EDA
        st.subheader("📈 Visualizaciones")
        crear_graficos_eda(data)

    with tab2:
        st.header("🤖 Entrenamiento de Modelos")

        with st.spinner("🔄 Entrenando modelos de ML..."):
            resultados, X_train, X_test = entrenar_modelos(data)

        st.success("✅ Modelos entrenados exitosamente!")

        # Información del entrenamiento
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Muestras de Entrenamiento", len(X_train))

        with col2:
            st.metric("Muestras de Prueba", len(X_test))

        with col3:
            st.metric("Características", X_train.shape[1])

    with tab3:
        st.header("📈 Métricas de Rendimiento")

        if "resultados" in locals():
            mostrar_metricas_modelo(resultados)
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Modelos ML'")

    with tab4:
        if "resultados" in locals():
            seccion_prediccion(resultados, data)
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Modelos ML'")


if __name__ == "__main__":
    main()