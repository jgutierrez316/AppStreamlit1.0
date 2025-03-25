import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Cargar el dataset Iris
data = sns.load_dataset('iris')

# Título de la aplicación
st.title("Análisis y Predicción de la Flor Iris")

# Sidebar para navegación
st.sidebar.title("Menú")
seccion = st.sidebar.radio("Selecciona una opción:", ["Exploración de Datos", "Visualización", "Predicción"])

if seccion == "Exploración de Datos":
    st.header("Exploración de Datos")
    st.write("Este conjunto de datos contiene información sobre tres especies de flores Iris.")
    st.dataframe(data)
    st.write("### Estadísticas Generales")
    st.write(data.describe())
    
    # Selección de especie
    especie = st.selectbox("Filtrar por especie:", data['species'].unique())
    st.write(data[data['species'] == especie])

elif seccion == "Visualización":
    st.header("Visualización de Datos")
    
    # Gráfico de dispersión
    st.write("### Relación entre características")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="sepal_length", y="sepal_width", hue="species", ax=ax)
    st.pyplot(fig)
    
    # Mapa de calor de correlaciones
    st.write("### Correlaciones entre características")
    fig, ax = plt.subplots()
    sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif seccion == "Predicción":
    st.header("Predicción de Especie de Iris")
    st.write("Ingrese las características de la flor para predecir su especie.")
    
    # Entradas de usuario
    sl = st.slider("Largo del sépalo", float(data.sepal_length.min()), float(data.sepal_length.max()), 5.0)
    sw = st.slider("Ancho del sépalo", float(data.sepal_width.min()), float(data.sepal_width.max()), 3.0)
    pl = st.slider("Largo del pétalo", float(data.petal_length.min()), float(data.petal_length.max()), 3.5)
    pw = st.slider("Ancho del pétalo", float(data.petal_width.min()), float(data.petal_width.max()), 1.0)
    
    # Preparar datos para el modelo
    X = data.drop(columns=['species'])
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimización: Cachear el modelo para evitar reentrenamiento
    @st.cache_resource
    def train_model():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    
    model = train_model()
    
    # Predicción
    if st.button("Predecir Especie"):
        try:
            datos = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
            resultado = model.predict(datos)
            st.success("Predicción realizada con éxito")
            st.write(f"🔍 **La especie predicha es:** `{resultado[0]}`")
        except Exception as e:
            st.error(f"⚠️ Error en la predicción: {e}")