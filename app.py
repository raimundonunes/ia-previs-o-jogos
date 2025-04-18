# Projeto: IA para Previsão de Jogos Virtuais (Futebol Virtual)

# ===================================
# Etapa 1: Gerar dados fictícios simulando histórico de jogos
# ===================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import streamlit as st
import joblib

# Gerar dados fictícios
def gerar_dados_ficticios(n=1000):
    data = []
    for _ in range(n):
        media_gols_mandante = round(random.uniform(0.5, 3.5), 2)
        media_gols_visitante = round(random.uniform(0.5, 3.5), 2)
        odd_mandante = round(random.uniform(1.5, 3.5), 2)
        odd_empate = round(random.uniform(2.5, 4.0), 2)
        odd_visitante = round(random.uniform(2.0, 4.5), 2)

        total = (1/odd_mandante) + (1/odd_empate) + (1/odd_visitante)
        prob_mandante = (1/odd_mandante) / total
        prob_empate = (1/odd_empate) / total
        prob_visitante = (1/odd_visitante) / total

        resultado = np.random.choice(['mandante', 'empate', 'visitante'], p=[prob_mandante, prob_empate, prob_visitante])

        data.append({
            'media_gols_mandante': media_gols_mandante,
            'media_gols_visitante': media_gols_visitante,
            'odd_mandante': odd_mandante,
            'odd_empate': odd_empate,
            'odd_visitante': odd_visitante,
            'resultado': resultado
        })
    return pd.DataFrame(data)

def treinar_ou_carregar_modelo():
    try:
        return joblib.load("modelo_jogos_virtuais.pkl")
    except:
        df = gerar_dados_ficticios()
        mapeamento = {'mandante': 1, 'empate': 0, 'visitante': -1}
        df['resultado_cod'] = df['resultado'].map(mapeamento)
        X = df[['media_gols_mandante', 'media_gols_visitante', 'odd_mandante', 'odd_empate', 'odd_visitante']]
        y = df['resultado_cod']
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        joblib.dump(modelo, "modelo_jogos_virtuais.pkl")
        return modelo

modelo = treinar_ou_carregar_modelo()

st.set_page_config(page_title="Previsor de Jogos Virtuais", layout="centered", page_icon="⚽")
with st.container():
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #00c853;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #00e676;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚽ IA para Previsão de Jogos Virtuais")
    st.markdown("""
    Preveja resultados de partidas virtuais com base em **média de gols** e **odds**. Ideal para simular estratégias no Futebol Virtual da Bet365!
    """)

    with st.form("form_previsao"):
        col1, col2 = st.columns(2)
        with col1:
            media_gols_m = st.slider("Média de Gols do Mandante", 0.5, 4.0, 2.0, 0.1)
            odd_m = st.number_input("Odd para Vitória do Mandante", min_value=1.0, max_value=10.0, value=1.90)
            odd_e = st.number_input("Odd para Empate", min_value=1.0, max_value=10.0, value=3.20)
        with col2:
            media_gols_v = st.slider("Média de Gols do Visitante", 0.5, 4.0, 2.0, 0.1)
            odd_v = st.number_input("Odd para Vitória do Visitante", min_value=1.0, max_value=10.0, value=3.60)

        submit = st.form_submit_button("🔮 Prever Resultado")

    if submit:
        entrada = pd.DataFrame([[media_gols_m, media_gols_v, odd_m, odd_e, odd_v]],
                               columns=['media_gols_mandante', 'media_gols_visitante', 'odd_mandante', 'odd_empate', 'odd_visitante'])
        pred = modelo.predict(entrada)[0]
        resultado_map = {1: '🏠 Vitória Mandante', 0: '🤝 Empate', -1: '🚌 Vitória Visitante'}
        st.success(f"Resultado Previsto: **{resultado_map[pred]}**")
        st.caption("Este modelo é baseado em simulações estatísticas. Use como ferramenta auxiliar!")
