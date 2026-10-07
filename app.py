import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator

st.set_page_config(page_title="Trading Confluence App", layout="wide", initial_sidebar_state="expanded")

# --- CATALOGO DE ACTIVOS ---
ASSETS = {
    "Acciones": {
        "CRCL": "CRCL", "COIN": "COIN", "MSTR": "MSTR", "LCID": "LCID", "ORCL": "ORCL",
        "NBIS": "NBIS", "UBER": "UBER", "NKE": "NKE", "GOOGL": "GOOGL", "NFLX": "NFLX",
        "MCD": "MCD", "AMD": "AMD", "TSLA": "TSLA", "INTC": "INTC", "PLTR": "PLTR",
        "AAPL": "AAPL", "MSFT": "MSFT", "AMZN": "AMZN", "AVGO": "AVGO", "KO": "KO",
        "CVX": "CVX", "TSM": "TSM", "HOOD": "HOOD", "CEG": "CEG", "SPCX": "SPCX"
    },
    "Criptomonedas": {
        "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "SOL/USD": "SOL-USD",
        "TRUMP/USD": "TRUMP-USD", "XRP/USD": "XRP-USD", "DOGE/USD": "DOGE-USD"
    },
    "Futuros": {
        "Oro": "GC=F", "Plata": "SI=F", "Cobre": "HG=F", "Platino": "PL=F",
        "Petróleo Crudo": "CL=F", "Gas Natural": "NG=F", "Maíz": "ZC=F", "Trigo": "ZW=F"
    },
    "Bonos": {
        "Bono EE.UU. 10Y": "^TNX"
    }
}

# --- FUNCIONES DE CÁLCULO TÉCNICO ---
def calculate_indicators(df):
    # Tendencias
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    
    # Osciladores
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Volatilidad
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Soportes y Resistencias locales (últimas 20 velas)
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    return df

def detect_candlestick_patterns(df):
    patterns = []
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    body = abs(last_row['Close'] - last_row['Open'])
    candle_range = last_row['High'] - last_row['Low']
    
    # Martillo (Hammer) / Martillo Invertido
    if candle_range > 0:
        lower_wick = min(last_row['Open'], last_row['Close']) - last_row['Low']
        upper_wick = last_row['High'] - max(last_row['Open'], last_row['Close'])
        
        if lower_wick > 2 * body and upper_wick < body:
            patterns.append(("Martillo (Alcista)", "Bullish"))
        if upper_wick > 2 * body and lower_wick < body:
            patterns.append(("Martillo Invertido (Alcista)", "Bullish"))
            patterns.append(("Estrella Fugaz (Bajista)", "Bearish"))
            
    # Envolventes
    if last_row['Close'] > last_row['Open'] and prev_row['Close'] < prev_row['Open']:
        if last_row['Close'] >= prev_row['Open'] and last_row['Open'] <= prev_row['Close']:
            patterns.append(("Envolvente Alcista", "Bullish"))
            
    if last_row['Close'] < last_row['Open'] and prev_row['Close'] > prev_row['Open']:
        if last_row['Close'] <= prev_row['Open'] and last_row['Open'] >= prev_row['Close']:
            patterns.append(("Envolvente Bajista", "Bearish"))
            
    return patterns

def evaluate_confluence(df):
    last = df.iloc[-1]
    bullish_signals = []
    bearish_signals = []
    
    # 1. Análisis de Tendencia
    if last['Close'] > last['SMA_200']:
        bullish_signals.append("Precio por encima de SMA 200 (Tendencia Principal Alcista)")
    else:
        bearish_signals.append("Precio por debajo de SMA 200 (Tendencia Principal Bajista)")
        
    if last['EMA_20'] > last['EMA_50']:
        bullish_signals.append("Cruce de medias rápidas EMA 20 > EMA 50 (Alcista)")
    else:
        bearish_signals.append("Cruce de medias rápidas EMA 20 < EMA 50 (Bajista)")
        
    # 2. Osciladores
    if last['RSI'] < 30:
        bullish_signals.append(f"RSI Sobrecomprado/Sobrevendido ({last['RSI']:.1f} < 30)")
    elif last['RSI'] > 70:
        bearish_signals.append(f"RSI Sobrecomprado ({last['RSI']:.1f} > 70)")
        
    if last['Stoch_K'] < 20 and last['Stoch_K'] > last['Stoch_D']:
        bullish_signals.append("Estocástico cruce alcista en zona de sobreventa")
    elif last['Stoch_K'] > 80 and last['Stoch_K'] < last['Stoch_D']:
        bearish_signals.append("Estocástico cruce bajista en zona de sobrecompra")

    # 3. Bandas de Bollinger
    if last['Close'] <= last['BB_Lower']:
        bullish_signals.append("Precio en la Banda Inferior de Bollinger (Rebote potencial)")
    elif last['Close'] >= last['BB_Upper']:
        bearish_signals.append("Precio en la Banda Superior de Bollinger (Resistencia)")
        
    # 4. Volumen
    avg_vol = df['Volume'].tail(20).mean()
    if last['Volume'] > 1.5 * avg_vol:
        bullish_signals.append("Volumen inusualmente alto confirmando movimiento")
        
    # 5. Patrones de velas
    patterns = detect_candlestick_patterns(df)
    for p_name, p_type in patterns:
        if p_type == "Bullish":
            bullish_signals.append(f"Patrón de vela: {p_name}")
        else:
            bearish_signals.append(f"Patrón de vela: {p_name}")

    # Puntuación de Confluencia
    total_checks = len(bullish_signals) + len(bearish_signals)
    if total_checks == 0:
        return "NEUTRAL", 0, [], [], 0, 0, 0

    bullish_score = (len(bullish_signals) / total_checks) * 100
    
    current_price = last['Close']
    if bullish_score >= 65:
        bias = "COMPRAR (BULLISH)"
        sl = last['Support'] * 0.995
        tp = current_price + (2 * (current_price - sl))
    elif bullish_score <= 35:
        bias = "VENTAR (BEARISH)"
        sl = last['Resistance'] * 1.005
        tp = current_price - (2 * (sl - current_price))
    else:
        bias = "NEUTRAL / ESPERAR"
        sl, tp = current_price, current_price
        
    return bias, bullish_score, bullish_signals, bearish_signals, current_price, sl, tp

# --- INTERFAZ STREAMLIT ---
st.title("📈 Confluence Trading Assistant")
st.caption("Sistema de análisis multinivel con confluencia técnica en tiempo real")

# Menú lateral para selección
st.sidebar.header("Configuración del Activo")
category = st.sidebar.selectbox("Categoría", list(ASSETS.keys()))
asset_name = st.sidebar.selectbox("Activo", list(ASSETS[category].keys()))
ticker = ASSETS[category][asset_name]
timeframe = st.sidebar.selectbox("Temporalidad", ["1d", "1h", "15m"], index=0)

# Descargar datos
data = yf.download(ticker, period="6m", interval=timeframe)

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    df = calculate_indicators(data)
    bias, score, bullish_reasons, bearish_reasons, entry, sl, tp = evaluate_confluence(df)
    
    # Métricas Principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Actual", f"${entry:.2f}")
    col2.metric("Dirección", bias)
    col3.metric("Confluencia Alcista", f"{score:.1f}%")
    col4.metric("Risk/Reward (Ratio)", "1:2")
    
    st.markdown("---")
    
    # Detalle de Operación
    if "COMPRAR" in bias or "VENTA" in bias:
        st.subheader("🎯 Plan de Trading Sugerido")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Precio de Entrada:** ${entry:.2f}")
        c2.error(f"**Stop Loss (SL):** ${sl:.2f}")
        c3.success(f"**Take Profit (TP):** ${tp:.2f}")
    
    # Desglose de Confluencias
    st.subheader("💡 Explicación de Confluencias Detectadas")
    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.write("🟢 **Factores Alcistas:**")
        for reason in bullish_reasons:
            st.write(f"- {reason}")
    with col_bear:
        st.write("🔴 **Factores Bajistas:**")
        for reason in bearish_reasons:
            st.write(f"- {reason}")

    # Gráfico Estilo TradingView
    st.subheader(f"📊 Gráfico Técnico: {asset_name}")
    fig = go.Figure()
    
    # Velas Japonésas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Precio"
    ))
    
    # Medias Móviles y Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1.5), name="SMA 200"))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dash'), name="Bollinger Sup."))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dash'), name="Bollinger Inf."))
    
    # Marcadores de Niveles Operativos
    if "COMPRAR" in bias or "VENTA" in bias:
        fig.add_hline(y=entry, line_dash="dash", line_color="blue", annotation_text="Entrada")
        fig.add_hline(y=sl, line_dash="dash", line_color="red", annotation_text="Stop Loss")
        fig.add_hline(y=tp, line_dash="dash", line_color="green", annotation_text="Take Profit")

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No se pudieron obtener datos para el activo seleccionado.")
