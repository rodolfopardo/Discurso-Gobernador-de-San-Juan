import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import io
import networkx as nx
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

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
palabras = re.findall(r'\b\w+\b', texto)
oraciones = sent_tokenize(texto)
st.subheader("📌 Indicadores Generales del Discurso")
kpi1, kpi2 = st.columns(2)
kpi1.metric("📝 Cantidad de Palabras", f"{len(palabras):,}")
kpi2.metric("🔠 Cantidad de Oraciones", f"{len(oraciones):,}")

# Limpieza y tokenización
stop_words = set(stopwords.words('spanish')).union({"sanjuan", "juan", "san"})
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

# Solo oraciones positivas
st.subheader("🟩 Top 5 Oraciones Más Positivas")
scored_sentences = [(frase.strip(), TextBlob(frase).sentiment.polarity) for frase in oraciones if len(frase.split()) > 4]
sorted_by_polarity = sorted(scored_sentences, key=lambda x: x[1])
positivas = sorted_by_polarity[-5:][::-1]

for frase, score in positivas:
    st.markdown(f"- *{frase}* (`{score:.2f}`)")

# Palabras clave con RAKE (frases con impacto)
st.subheader("💥 Frases más impactantes del discurso")
st.markdown("Estas frases han sido extraídas automáticamente por su relevancia e impacto en el texto. Ayudan a entender los mensajes más destacados y repetidos.")
r = Rake(language='spanish')
r.extract_keywords_from_text(texto)
keywords = r.get_ranked_phrases_with_scores()[:20]
df_keywords = pd.DataFrame(keywords, columns=["Impacto", "Frase clave"])
st.dataframe(df_keywords)

# Red semántica
st.subheader("🔗 Red Semántica de Conceptos Clave")
st.markdown("Una red semántica muestra la relación entre palabras que aparecen juntas frecuentemente en el discurso. Es útil para visualizar los temas más conectados entre sí.")

pairs = list(zip(palabras_filtradas[:-1], palabras_filtradas[1:]))
coocurrencias = Counter(pairs)
top_pairs = coocurrencias.most_common(30)

G = nx.Graph()
for (a, b), w in top_pairs:
    G.add_edge(a, b, weight=w)

pos = nx.spring_layout(G, k=0.5, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", edge_color="gray", font_size=10, width=[G[u][v]['weight']/2 for u,v in G.edges()])
st.pyplot(plt.gcf())

# Resumen automático
st.subheader("📝 Resumen automático del discurso")
st.markdown("Utilizamos el modelo LSA para generar un resumen automático del contenido.")
parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
summarizer = LsaSummarizer()
resumen = summarizer(parser.document, 5)

for i, oracion in enumerate(resumen, 1):
    st.markdown(f"**{i}.** {oracion}")

# Descarga del análisis como CSV
st.markdown("---")
output_csv = df_keywords.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar frases clave como CSV", data=output_csv, file_name="frases_clave.csv", mime="text/csv")
