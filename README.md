# 🛒 Predictor de Compras - ML Demo

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 👨‍🎓 Información del Proyecto

**Estudiante:** Maximiliano Bustamante  
**Curso:** Inteligencia Artificial  
**Universidad:** EAFIT  
**Fecha:** 20 de Agosto, 2025  

---

## 📋 Descripción del Proyecto

Esta aplicación web desarrollada con **Streamlit** implementa un modelo supervisado de **Machine Learning** para predecir la probabilidad de compra de clientes basándose en sus características demográficas y comportamentales.

### 🎯 Objetivo
Demostrar la implementación completa de un pipeline de ML que incluye:
- Generación de datos sintéticos
- Análisis exploratorio de datos (EDA)
- Entrenamiento de modelos supervisados
- Evaluación de métricas de rendimiento
- Interfaz interactiva para predicciones

---

## 🔧 Características Técnicas

### Dataset Sintético
- **📊 Muestras:** 500+ registros (configurable: 300-1000)
- **📈 Variables:** 6 características + 1 variable target
- **🎯 Tipo de problema:** Clasificación binaria
- **🔄 Generación:** Datos sintéticos con lógica de negocio realista

### Variables del Dataset
| Variable | Descripción | Tipo |
|----------|-------------|------|
| `edad` | Edad del cliente (18-80 años) | Numérica |
| `ingresos_anuales` | Ingresos anuales del cliente | Numérica |
| `tiempo_web_minutos` | Tiempo navegando en el sitio web | Numérica |
| `productos_vistos` | Cantidad de productos visualizados | Numérica |
| `historial_compras` | Número de compras previas | Numérica |
| `puntuacion_credito` | Puntuación crediticia (300-850) | Numérica |
| `compra` | Variable target (0=No compra, 1=Compra) | Binaria |

### Modelos Implementados
1. **🌲 Random Forest Classifier**
   - N_estimators: 100
   - Max_depth: 10
   - Random_state: 42

2. **📊 Regresión Logística**
   - Regularización L2
   - Max_iter: 1000
   - Escalado con StandardScaler

### Métricas de Evaluación
- ✅ **Accuracy:** Precisión general del modelo
- 🎯 **Precision:** Precisión de predicciones positivas
- 🔍 **Recall:** Capacidad de detectar casos positivos
- ⚖️ **F1-Score:** Media armónica entre precision y recall
- 📊 **Matriz de Confusión:** Visualización de predicciones vs realidad

---

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 1️⃣ Clonar o Descargar el Proyecto
```bash
# Si tienes git instalado
git clone <url-del-repositorio>
cd predictor-compras-ml

# O simplemente descarga los archivos main.py y requirements.txt
```

### 2️⃣ Crear Entorno Virtual (Recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecutar la Aplicación
```bash
streamlit run main.py
```

La aplicación estará disponible en: `http://localhost:8501`

---

## 📱 Uso de la Aplicación

### 🏠 Navegación Principal
La aplicación está organizada en **4 pestañas principales**:

#### 1. 📊 **Exploración de Datos**
- **Estadísticas descriptivas** del dataset
- **Muestra de datos** para inspección
- **Visualizaciones interactivas:**
  - Distribución de la variable target
  - Matriz de correlación entre variables
  - Distribución de edad por compra
  - Relación ingresos vs tiempo en web

#### 2. 🤖 **Modelos ML**
- **Entrenamiento automático** de modelos
- **División train/test** (80/20)
- **Información del proceso:**
  - Número de muestras de entrenamiento
  - Número de muestras de prueba
  - Cantidad de características

#### 3. 📈 **Métricas**
- **Métricas de rendimiento** para cada modelo
- **Visualización de matrices de confusión**
- **Comparación entre modelos**

#### 4. 🔮 **Predicción**
- **Interfaz interactiva** con sliders
- **Predicción en tiempo real**
- **Probabilidades de predicción**
- **Visualización de resultados**

### ⚙️ Panel Lateral (Sidebar)
- **Información del proyecto**
- **Control del tamaño del dataset** (300-1000 muestras)
- **Descripción de variables**

---

## 📁 Estructura del Proyecto

```
predictor-compras-ml/
│
├── main.py                 # Aplicación principal de Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Documentación del proyecto
│
└── [generados automáticamente]
    ├── .streamlit/        # Configuraciones de Streamlit
    └── __pycache__/       # Cache de Python
```

---

## 🧩 Arquitectura del Código

### Funciones Principales

#### `generar_datos_simulados(n_samples)`
```python
"""
Genera dataset sintético con lógica de negocio coherente.
- Distribuciones realistas para cada variable
- Variable target basada en características del cliente
- Correlaciones lógicas entre variables
"""
```

#### `entrenar_modelos(data)`
```python
"""
Pipeline completo de entrenamiento:
- División train/test estratificada
- Escalado de características (cuando necesario)
- Entrenamiento de múltiples modelos
- Cálculo de métricas de evaluación
"""
```

#### `crear_graficos_eda(data)`
```python
"""
Análisis exploratorio visual:
- Gráficos interactivos con Plotly
- Visualizaciones estadísticas con Seaborn
- Matrices de correlación
"""
```

#### `seccion_prediccion(resultados, data)`
```python
"""
Interfaz interactiva para predicciones:
- Controles deslizantes para características
- Predicción en tiempo real
- Visualización de probabilidades
"""
```

---

## 📊 Resultados Esperados

### Rendimiento de Modelos
Los modelos típicamente logran:
- **Accuracy:** ~75-85%
- **Precision:** ~70-80%
- **Recall:** ~70-85%
- **F1-Score:** ~70-82%

### Interpretación de Variables
**Variables más influyentes** (basado en la lógica de generación):
1. 💰 **Ingresos anuales** (30% de peso)
2. ⏱️ **Tiempo en web** (20% de peso)
3. 👁️ **Productos vistos** (20% de peso)
4. 📚 **Historial de compras** (15% de peso)
5. 👤 **Edad** (10% de peso)
6. 💳 **Puntuación crediticia** (5% de peso)

---

## 🔧 Personalización y Extensiones

### Modificar el Dataset
```python
# En la función generar_datos_simulados()
# Cambiar distribuciones:
edad = np.random.normal(40, 15, n_samples)  # Media 40, std 15
ingresos = np.random.lognormal(10.5, 0.6, n_samples)  # Mayores ingresos
```

### Agregar Nuevos Modelos
```python
# En la función entrenar_modelos()
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

modelos = {
    "Random Forest": RandomForestClassifier(...),
    "Regresión Logística": LogisticRegression(...),
    "SVM": SVC(probability=True, random_state=42),  # Nuevo modelo
    "Gradient Boosting": GradientBoostingClassifier(...)  # Nuevo modelo
}
```

### Modificar la Interfaz
```python
# Cambiar tema de colores, layout, o agregar nuevas visualizaciones
st.set_page_config(
    page_title="Mi Proyecto ML",
    page_icon="🤖",
    layout="centered",  # Cambiar a centered
)
```

---

## 🐛 Solución de Problemas

### Error: Módulo no encontrado
```bash
# Asegúrate de tener el entorno virtual activado
pip install -r requirements.txt
```

### Error: Puerto en uso
```bash
# Usa un puerto diferente
streamlit run main.py --server.port 8502
```

### Rendimiento lento
- Reduce el número de muestras en el sidebar
- Cierra otras aplicaciones que consuman recursos

---

## 🤝 Contribuciones y Mejoras

### Ideas para Extensiones Futuras
- 🔄 **Validación cruzada** para evaluación más robusta
- 🎛️ **Hyperparameter tuning** automático
- 📥 **Carga de datasets externos** (CSV)
- 🔍 **Interpretabilidad** con SHAP values
- 📊 **Más visualizaciones** (ROC curves, feature importance)
- 💾 **Persistencia de modelos** (pickle/joblib)
- 🌐 **Deploy en la nube** (Streamlit Cloud, Heroku)

---

## 📚 Referencias y Recursos

### Documentación Oficial
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python Documentation](https://plotly.com/python/)

### Cursos y Tutoriales
- [Curso de Inteligencia Artificial - EAFIT](https://www.eafit.edu.co/)
- [Machine Learning Course - Coursera](https://www.coursera.org/learn/machine-learning)

---

## 📄 Licencia

Este proyecto es desarrollado con fines educativos para el curso de Inteligencia Artificial en EAFIT.

---

## 👨‍💻 Autor

**Maximiliano Bustamante**  
📧 Email: [tu-email@eafit.edu.co]  
🎓 Universidad EAFIT  
📅 Agosto 2025  

---

*Proyecto desarrollado como parte del curso de Inteligencia Artificial en la Universidad EAFIT. Demuestra la implementación completa de un pipeline de Machine Learning con interfaz web interactiva.*