
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# Funções extraídas do notebook
def capturar_parametros(ticker, periodo='1y'):
    dados = yf.download(ticker, period=periodo)
    dados['Retornos'] = np.log(dados['Close'] / dados['Close'].shift(1))
    dados = dados.dropna()
    S0 = dados['Close'].iloc[-1].item()
    mu = dados['Retornos'].mean() * 252
    sigma = dados['Retornos'].std() * np.sqrt(252)
    return S0, mu, sigma, dados

def monte_carlo_opcao_europeia(S0, K, T, r, sigma, n_sim=10000):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# Interface Streamlit
st.title("Calculadora de Opções Europeias")
ticker = st.text_input("Código do ativo (ex: AAPL)", "AAPL")
K = st.number_input("Preço de exercício (Strike)", value=150.0)
T = st.number_input("Tempo até o vencimento (em anos)", value=1.0)
r = st.number_input("Taxa de juros livre de risco (anual)", value=0.05)
n_sim = st.slider("Número de simulações", 1000, 50000, 10000)

if st.button("Calcular"):
    with st.spinner("Calculando..."):
        S0, mu, sigma, dados = capturar_parametros(ticker)
        preco = monte_carlo_opcao_europeia(S0, K, T, r, sigma, n_sim)
        st.success(f"Preço estimado da opção europeia de compra: ${preco:.2f}")
