"""
Aplicación de Machine Learning con Streamlit
===========================================

Esta aplicación demuestra un modelo supervisado de clasificación para predecir
la compra de productos basado en características del cliente.

Autor: Maximiliano Bustamante
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
import io

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


def validar_csv(df):
    """
    Valida que el CSV tenga las columnas correctas.

    Args:
        df (pd.DataFrame): DataFrame a validar

    Returns:
        tuple: (bool, str) - (es_valido, mensaje)
    """
    columnas_requeridas = [
        "edad",
        "ingresos_anuales",
        "tiempo_web_minutos",
        "productos_vistos",
        "historial_compras",
        "puntuacion_credito",
        "compra",
    ]

    columnas_faltantes = set(columnas_requeridas) - set(df.columns)

    if columnas_faltantes:
        return False, f"Columnas faltantes: {', '.join(columnas_faltantes)}"

    # Verificar que 'compra' sea binaria
    if not df["compra"].isin([0, 1]).all():
        return (
            False,
            "La columna 'compra' debe contener solo valores 0 (no compra) y 1 (compra)",
        )

    # Verificar tipos de datos numéricos
    for col in columnas_requeridas[:-1]:  # Todas excepto 'compra'
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"La columna '{col}' debe ser numérica"

    return True, "CSV válido"


def cargar_datos_csv():
    """
    Interfaz para cargar datos desde CSV.

    Returns:
        pd.DataFrame: DataFrame cargado o None
    """
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV", 
        type="csv", 
        help="El archivo debe tener las columnas requeridas"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Mostrar información básica
            st.info(f"📊 Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Validar estructura
            es_valido, mensaje = validar_csv(df)
            
            if es_valido:
                st.success("✅ " + mensaje)
                
                # Mostrar vista previa
                st.subheader("👀 Vista Previa de los Datos")
                st.dataframe(df.head())
                
                # Estadísticas básicas
                st.subheader("📊 Estadísticas Básicas")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Distribución de la variable target:**")
                    target_counts = df["compra"].value_counts()
                    st.write(f"- No compra (0): {target_counts.get(0, 0)}")
                    st.write(f"- Compra (1): {target_counts.get(1, 0)}")
                
                with col2:
                    st.write("**Información del dataset:**")
                    st.write(f"- Total de registros: {len(df)}")
                    st.write(f"- Valores nulos: {df.isnull().sum().sum()}")
                
                return df
            else:
                st.error("❌ " + mensaje)
                st.info("El CSV debe tener estas columnas exactas:")
                columnas_ejemplo = [
                    "edad", "ingresos_anuales", "tiempo_web_minutos",
                    "productos_vistos", "historial_compras", "puntuacion_credito", "compra"
                ]
                st.code(", ".join(columnas_ejemplo))
                
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
    
    return None


@st.cache_data
def entrenar_modelos(data, modelos_seleccionados):
    """
    Entrena múltiples modelos de ML seleccionados y devuelve métricas.

    Args:
        data (pd.DataFrame): Dataset para entrenar
        modelos_seleccionados (list): Lista de nombres de modelos a entrenar

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

    # Definir todos los modelos disponibles
    todos_los_modelos = {
        "Random Forest": {
            "modelo": RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            ),
            "requiere_escalado": False,
            "tiene_arbol": True,
        },
        "Regresión Logística": {
            "modelo": LogisticRegression(random_state=42, max_iter=1000),
            "requiere_escalado": True,
            "tiene_arbol": False,
        },
        "SVM": {
            "modelo": SVC(random_state=42, probability=True, kernel="rbf"),
            "requiere_escalado": True,
            "tiene_arbol": False,
        },
        "Gradient Boosting": {
            "modelo": GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=6
            ),
            "requiere_escalado": False,
            "tiene_arbol": True,
        },
        "Decision Tree": {
            "modelo": DecisionTreeClassifier(
                random_state=42, max_depth=8, min_samples_split=10, min_samples_leaf=5
            ),
            "requiere_escalado": False,
            "tiene_arbol": True,
        },
    }

    # Filtrar solo los modelos seleccionados
    modelos_a_entrenar = {
        nombre: config
        for nombre, config in todos_los_modelos.items()
        if nombre in modelos_seleccionados
    }

    resultados = {}

    for nombre, config in modelos_a_entrenar.items():
        modelo = config["modelo"]
        requiere_escalado = config["requiere_escalado"]
        tiene_arbol = config["tiene_arbol"]

        # Entrenar el modelo
        if requiere_escalado:
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
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        resultados[nombre] = {
            "modelo": modelo,
            "metricas": metricas,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "scaler": scaler if requiere_escalado else None,
            "tiene_arbol": tiene_arbol,
            "feature_names": list(X.columns),
        }

    return resultados, X_train, X_test


def visualizar_arbol_decision(modelo, nombre_modelo, feature_names, max_depth_viz=4):
    """
    Visualiza el árbol de decisión de un modelo.

    Args:
        modelo: Modelo entrenado
        nombre_modelo (str): Nombre del modelo
        feature_names (list): Nombres de las características
        max_depth_viz (int): Profundidad máxima para visualización
    """
    st.subheader(f"🌳 Visualización del Árbol - {nombre_modelo}")
    
    # Configurar el tamaño de la visualización
    col1, col2 = st.columns([3, 1])
    
    with col2:
        max_depth_viz = st.slider(
            "Profundidad máxima:", 
            min_value=2, 
            max_value=10, 
            value=4, 
            key=f"depth_{nombre_modelo}"
        )
        
        show_text = st.checkbox(
            "Mostrar reglas de texto", 
            value=False, 
            key=f"text_{nombre_modelo}"
        )
    
    with col1:
        try:
            if nombre_modelo == "Decision Tree":
                # Para Decision Tree, mostrar el árbol directamente
                arbol_a_mostrar = modelo
            elif nombre_modelo == "Random Forest":
                # Para Random Forest, mostrar el primer estimador
                arbol_a_mostrar = modelo.estimators_[0]
                st.info("📊 Mostrando el primer árbol del Random Forest (de 100 total)")
            elif nombre_modelo == "Gradient Boosting":
                # Para Gradient Boosting, mostrar el primer estimador
                arbol_a_mostrar = modelo.estimators_[0, 0]
                st.info("📊 Mostrando el primer árbol del Gradient Boosting (de 100 total)")
            else:
                st.warning(f"⚠️ {nombre_modelo} no tiene visualización de árbol disponible")
                return
            
            # Crear la visualización
            fig, ax = plt.subplots(figsize=(20, 12))
            
            plot_tree(
                arbol_a_mostrar,
                feature_names=feature_names,
                class_names=['No Compra', 'Compra'],
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=max_depth_viz,
                ax=ax
            )
            
            plt.title(f"Árbol de Decisión - {nombre_modelo}\n(Profundidad máx: {max_depth_viz})", 
                     fontsize=16, fontweight='bold')
            st.pyplot(fig)
            
            # Mostrar reglas de texto si se solicita
            if show_text:
                st.subheader("📝 Reglas del Árbol en Texto")
                
                with st.expander("Ver reglas completas"):
                    reglas_texto = export_text(
                        arbol_a_mostrar,
                        feature_names=feature_names,
                        max_depth=max_depth_viz
                    )
                    st.text(reglas_texto)
            
            # Mostrar importancia de características para árboles individuales
            if hasattr(arbol_a_mostrar, 'feature_importances_'):
                st.subheader("📊 Importancia de Características")
                
                importancias = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': arbol_a_mostrar.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                fig_imp = px.bar(
                    importancias,
                    x='Importancia',
                    y='Característica',
                    orientation='h',
                    title=f"Importancia de Características - {nombre_modelo}"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Mostrar tabla de importancias
                st.dataframe(importancias.round(4))
                
        except Exception as e:
            st.error(f"Error al visualizar el árbol: {str(e)}")


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
    # Crear comparación de métricas
    if len(resultados) > 1:
        st.subheader("📊 Comparación de Modelos")
        
        metricas_df = pd.DataFrame({
            nombre: {
                "Accuracy": resultado["metricas"]["accuracy"],
                "Precision": resultado["metricas"]["precision"],
                "Recall": resultado["metricas"]["recall"],
                "F1-Score": resultado["metricas"]["f1"],
            }
            for nombre, resultado in resultados.items()
        }).round(3)
        
        st.dataframe(metricas_df)
        
        # Gráfico de barras comparativo
        fig_comp = px.bar(
            metricas_df.T.reset_index(),
            x="index",
            y=["Accuracy", "Precision", "Recall", "F1-Score"],
            title="Comparación de Métricas por Modelo",
            barmode="group"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Detalles individuales de cada modelo
    for nombre, resultado in resultados.items():
        st.subheader(f"📊 Métricas Detalladas - {nombre}")

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

    # Sidebar con configuraciones
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Seleccionar fuente de datos
        st.subheader("📊 Fuente de Datos")
        fuente_datos = st.radio(
            "Elige la fuente de datos:",
            ["Datos Simulados", "Cargar CSV"]
        )
        
        data = None  # Inicializar data
        
        if fuente_datos == "Datos Simulados":
            n_samples = st.slider("Número de muestras:", 300, 1000, 500)
            data = generar_datos_simulados(n_samples)
            st.success(f"✅ Usando datos simulados ({n_samples} muestras)")
        else:
            st.markdown("**Formato requerido del CSV:**")
            st.code("edad,ingresos_anuales,tiempo_web_minutos,productos_vistos,historial_compras,puntuacion_credito,compra")
            data_csv = cargar_datos_csv()
            
            if data_csv is not None:
                data = data_csv
                st.success("✅ Usando datos del CSV cargado")
            else:
                st.warning("⚠️ No se ha cargado ningún CSV. Usando datos simulados por defecto.")
                n_samples = st.slider("Número de muestras:", 300, 1000, 500, key="fallback_samples")
                data = generar_datos_simulados(n_samples)
        
        # Seleccionar modelos
        st.subheader("🤖 Selección de Modelos")
        modelos_disponibles = [
            "Random Forest",
            "Regresión Logística", 
            "SVM",
            "Gradient Boosting",
            "Decision Tree"
        ]
        
        modelos_seleccionados = []
        for modelo in modelos_disponibles:
            default_selected = modelo in ["Random Forest", "Regresión Logística", "Decision Tree"]
            if st.checkbox(modelo, value=default_selected):
                modelos_seleccionados.append(modelo)
        
        if not modelos_seleccionados:
            st.warning("⚠️ Selecciona al menos un modelo")
        
        # Información del proyecto
        st.markdown("---")
        st.header("ℹ️ Información del Proyecto")
        
        # Determinar el tipo de datos que se está usando
        tipo_datos = "CSV cargado" if fuente_datos == "Cargar CSV" and data is not None and st.session_state.get('csv_cargado', False) else "Simulados"
        
        st.markdown(
            f"""
        **Dataset Actual:**
        - 📊 Tipo: {tipo_datos}
        - 🔢 Muestras: {len(data) if data is not None else 0}
        - 📊 Variables: 6 características + 1 target
        - 🎯 Tipo: Clasificación binaria
        - 🤖 Modelos seleccionados: {len(modelos_seleccionados)}
        
        **Variables:**
        - Edad del cliente
        - Ingresos anuales
        - Tiempo en sitio web
        - Productos vistos
        - Historial de compras
        - Puntuación crediticia
        """
        )

    # Verificar que hay modelos seleccionados (data siempre existirá ahora)
    if not modelos_seleccionados:
        st.error("❌ Selecciona al menos un modelo en la barra lateral")
        st.stop()

    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Exploración de Datos", "🤖 Modelos ML", "📈 Métricas", "🌳 Árboles", "🔮 Predicción"]
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
        
        st.info(f"🎯 Modelos seleccionados: {', '.join(modelos_seleccionados)}")

        with st.spinner("🔄 Entrenando modelos de ML..."):
            resultados, X_train, X_test = entrenar_modelos(data, modelos_seleccionados)

        st.success("✅ Modelos entrenados exitosamente!")

        # Información del entrenamiento
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Muestras de Entrenamiento", len(X_train))

        with col2:
            st.metric("Muestras de Prueba", len(X_test))

        with col3:
            st.metric("Características", X_train.shape[1])
            
        with col4:
            st.metric("Modelos Entrenados", len(resultados))

    with tab3:
        st.header("📈 Métricas de Rendimiento")

        if "resultados" in locals() and resultados:
            mostrar_metricas_modelo(resultados)
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Modelos ML'")

    with tab4:
        st.header("🌳 Visualización de Árboles de Decisión")
        
        if "resultados" in locals() and resultados:
            # Filtrar modelos que tienen árboles
            modelos_con_arboles = {
                nombre: resultado 
                for nombre, resultado in resultados.items() 
                if resultado.get("tiene_arbol", False)
            }
            
            if modelos_con_arboles:
                # Selector de modelo para visualizar
                modelo_arbol = st.selectbox(
                    "Selecciona el modelo para visualizar:",
                    list(modelos_con_arboles.keys()),
                    key="selector_arbol"
                )
                
                if modelo_arbol:
                    resultado_modelo = modelos_con_arboles[modelo_arbol]
                    visualizar_arbol_decision(
                        resultado_modelo["modelo"],
                        modelo_arbol,
                        resultado_modelo["feature_names"]
                    )
            else:
                st.warning("⚠️ No hay modelos con árboles de decisión seleccionados. Selecciona Decision Tree, Random Forest o Gradient Boosting.")
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Modelos ML'")

    with tab5:
        if "resultados" in locals() and resultados:
            seccion_prediccion(resultados, data)
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Modelos ML'")


if __name__ == "__main__":
    main()