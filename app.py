import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import spacy
from collections import Counter
from nltk.corpus import stopwords
from rake_nltk import Rake
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import io
import networkx as nx
import numpy as np
import nltk

nltk.download('punkt')
try:
    stop_words = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))

# Configurar autenticación simple
USER = "daniel"
PASSWORD = "sanjuan"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🗣️ Análisis del Discurso de Apertura de Sesiones")
    user = st.text_input("Usuario")
    passwd = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        if user == USER and passwd == PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Usuario o contraseña incorrectos")
    st.stop()

# Si pasa la autenticación
st.title("📘 Análisis del Discurso de Apertura de Sesiones")

# Cargar texto
def load_text():
    with open("discurso_gobernador.txt", "r", encoding="utf-8") as f:
        return f.read()

texto = load_text()
st.subheader("📄 Discurso cargado")
st.text_area("Contenido", texto[:3000] + "...", height=300)

# KPIs básicos
from nltk.tokenize import sent_tokenize
palabras = re.findall(r'\b\w+\b', texto)
oraciones = sent_tokenize(texto)
st.subheader("📌 Indicadores Generales del Discurso")
kpi1, kpi2 = st.columns(2)
kpi1.metric("📝 Cantidad de Palabras", f"{len(palabras):,}")
kpi2.metric("🔠 Cantidad de Oraciones", f"{len(oraciones):,}")

# Limpieza y tokenización
stop_words = stop_words.union({"sanjuan", "juan", "san"})
palabras_filtradas = [p.lower() for p in palabras if p.lower() not in stop_words and len(p) > 3]

# WordCloud interactivo como imagen descargable
st.subheader("☁️ Nube de Palabras")
wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(" ".join(palabras_filtradas))
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

img_buf = io.BytesIO()
wc.to_image().save(img_buf, format='PNG')
st.download_button("📥 Descargar WordCloud", data=img_buf.getvalue(), file_name="nube_palabras.png", mime="image/png")

# Frecuencia de palabras con gráfico de barras
st.subheader("📊 Palabras más frecuentes")
conteo = Counter(palabras_filtradas).most_common(20)
df_frecuencias = pd.DataFrame(conteo, columns=["Palabra", "Frecuencia"])
fig_bar = px.bar(df_frecuencias, x="Palabra", y="Frecuencia", title="Top 20 Palabras Frecuentes", color="Frecuencia", template="plotly_white")
st.plotly_chart(fig_bar, use_container_width=True)

# Sentimiento general con interpretación y visual
st.subheader("💬 Análisis de Sentimiento General")
sentimiento = TextBlob(texto)
polaridad = sentimiento.sentiment.polarity
subjetividad = sentimiento.sentiment.subjectivity

if polaridad > 0.2:
    interpretacion = "El discurso tiene un tono mayormente positivo, transmitiendo esperanza, logros y visión de futuro."
elif polaridad < -0.2:
    interpretacion = "El discurso tiene un tono mayormente negativo, con un enfoque crítico o en problemas."
else:
    interpretacion = "El discurso mantiene un tono neutral, equilibrando logros y dificultades."

# Gráfico de polaridad en escala
fig_sent = go.Figure()
fig_sent.add_trace(go.Indicator(
    mode="gauge+number",
    value=polaridad,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Polaridad del Discurso"},
    gauge={
        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [-1, -0.2], 'color': 'red'},
            {'range': [-0.2, 0.2], 'color': 'lightgray'},
            {'range': [0.2, 1], 'color': 'green'}
        ],
    }))
fig_sent.update_layout(height=300)
st.plotly_chart(fig_sent, use_container_width=True)
st.metric("Subjetividad", f"{subjetividad:.2f}", help="Valores cercanos a 1 indican alta carga emocional u opinativa")
st.markdown(f"**🧠 Interpretación:** {interpretacion}")

# Top 5 oraciones más positivas
st.subheader("🟩 Top 5 Oraciones con Sentimiento Positivo")
scored_sentences = [(oracion, TextBlob(oracion).sentiment.polarity) for oracion in oraciones if len(oracion.split()) > 4]
sorted_positive = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
top_positivas = [s for s in sorted_positive if s[1] > 0][:5]

texto_positivas = ""
for i, (frase, score) in enumerate(top_positivas, 1):
    st.markdown(f"**{i}.** *{frase.strip()}* (`{score:.2f}`)")
    texto_positivas += f"{i}. {frase.strip()} ({score:.2f})\n"

st.download_button("📥 Descargar oraciones positivas", data=texto_positivas, file_name="oraciones_positivas.txt", mime="text/plain")

# Resumen automático alternativo simple (basado en frecuencia)
st.subheader("📝 Resumen automático del discurso")
st.markdown("Utilizamos una técnica basada en frecuencia para extraer oraciones representativas del contenido.")

frecuencia = Counter(palabras_filtradas)
sentencias_con_puntaje = []
for oracion in oraciones:
    puntaje = sum(frecuencia.get(p.lower(), 0) for p in re.findall(r'\b\w+\b', oracion))
    sentencias_con_puntaje.append((oracion, puntaje))

resumen_top = sorted(sentencias_con_puntaje, key=lambda x: x[1], reverse=True)[:5]
for i, (oracion, puntaje) in enumerate(resumen_top, 1):
    st.markdown(f"**{i}.** {oracion.strip()}")
