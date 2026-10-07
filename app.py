import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator

st.set_page_config(page_title="Trading Confluence & Divergence App", layout="wide", initial_sidebar_state="expanded")

# --- CATÁLOGO DE ACTIVOS ---
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

# MAPA DE TEMPORALIDADES Y PERÍODOS DE YFINANCE
TIMEFRAME_CONFIG = {
    "5 Minutos (5m)": {"interval": "5m", "period": "5d"},
    "15 Minutos (15m)": {"interval": "15m", "period": "1mo"},
    "30 Minutos (30m)": {"interval": "30m", "period": "1mo"},
    "1 Hora (1h)": {"interval": "1h", "period": "2mo"},
    "4 Horas (4h)": {"interval": "60m", "period": "3mo"},  # Agrupado manualmente
    "1 Día (1d)": {"interval": "1d", "period": "1y"},
    "1 Semana (1wk)": {"interval": "1wk", "period": "3y"},
    "1 Mes (1mo)": {"interval": "1mo", "period": "5y"}
}

# --- CÁLCULOS TÉCNICOS ---
def calculate_indicators(df):
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    return df

# DETECCIÓN DE PATRONES DE VELAS EN MÚLTIPLES VELAS RECIENTES
def detect_all_candlestick_patterns(df, lookback=40):
    annotations = []
    start_idx = max(2, len(df) - lookback)
    
    for i in range(start_idx, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        body = abs(curr['Close'] - curr['Open'])
        candle_range = curr['High'] - curr['Low']
        if candle_range == 0:
            continue
            
        lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
        upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
        
        date_val = df.index[i]
        
        # Martillo (Bullish)
        if lower_wick > 2 * body and upper_wick < body:
            annotations.append((date_val, curr['Low'], "🔨 Martillo", "green", "bottom"))
        # Estrella Fugaz (Bearish)
        elif upper_wick > 2 * body and lower_wick < body:
            annotations.append((date_val, curr['High'], "💫 Estrella Fugaz", "red", "top"))
        # Envolvente Alcista
        elif curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and curr['Close'] >= prev['Open'] and curr['Open'] <= prev['Close']:
            annotations.append((date_val, curr['Low'], "🟢 Env. Alcista", "green", "bottom"))
        # Envolvente Bajista
        elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and curr['Close'] <= prev['Open'] and curr['Open'] >= prev['Close']:
            annotations.append((date_val, curr['High'], "🔴 Env. Bajista", "red", "top"))
            
    return annotations

# DETECCIÓN DE DIVERGENCIAS ALCISTAS Y BAJISTAS (RSI VS PRECIO)
def detect_rsi_divergences(df, window=5, lookback=60):
    df_sub = df.tail(lookback).copy()
    bullish_divs = []
    bearish_divs = []
    
    # Encontrar pivotes mínimos (Soportes locales) y máximos (Resistencias locales)
    lows = df_sub['Low'].values
    highs = df_sub['High'].values
    rsi = df_sub['RSI'].values
    dates = df_sub.index
    
    pivot_lows = []
    pivot_highs = []
    
    for i in range(window, len(df_sub) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            pivot_lows.append(i)
        if highs[i] == max(highs[i-window:i+window+1]):
            pivot_highs.append(i)
            
    # Comparar pivotes mínimos para Divergencia Alcista (Precio baja, RSI sube)
    for k in range(1, len(pivot_lows)):
        idx1, idx2 = pivot_lows[k-1], pivot_lows[k]
        if lows[idx2] < lows[idx1] and rsi[idx2] > rsi[idx1]:
            bullish_divs.append({
                'p1_date': dates[idx1], 'p1_price': lows[idx1], 'p1_rsi': rsi[idx1],
                'p2_date': dates[idx2], 'p2_price': lows[idx2], 'p2_rsi': rsi[idx2]
            })
            
    # Comparar pivotes máximos para Divergencia Bajista (Precio sube, RSI baja)
    for k in range(1, len(pivot_highs)):
        idx1, idx2 = pivot_highs[k-1], pivot_highs[k]
        if highs[idx2] > highs[idx1] and rsi[idx2] < rsi[idx1]:
            bearish_divs.append({
                'p1_date': dates[idx1], 'p1_price': highs[idx1], 'p1_rsi': rsi[idx1],
                'p2_date': dates[idx2], 'p2_price': highs[idx2], 'p2_rsi': rsi[idx2]
            })
            
    return bullish_divs, bearish_divs

# EVALUACIÓN GENERAL DE CONFLUENCIA
def evaluate_confluence(df, bull_divs, bear_divs):
    last = df.iloc[-1]
    bullish_signals, bearish_signals = [], []
    
    if last['Close'] > last['SMA_200']:
        bullish_signals.append("Precio sobre SMA 200 (Tendencia Alcista)")
    else:
        bearish_signals.append("Precio bajo SMA 200 (Tendencia Bajista)")
        
    if last['EMA_20'] > last['EMA_50']:
        bullish_signals.append("Cruce de Medias Rápido (EMA 20 > EMA 50)")
    else:
        bearish_signals.append("Cruce de Medias Bajista (EMA 20 < EMA 50)")
        
    if last['RSI'] < 30:
        bullish_signals.append(f"RSI en Zona de Sobreventa ({last['RSI']:.1f})")
    elif last['RSI'] > 70:
        bearish_signals.append(f"RSI en Zona de Sobrecompra ({last['RSI']:.1f})")
        
    if len(bull_divs) > 0:
        bullish_signals.append("🔥 DIVERGENCIA ALCISTA detectada en el RSI")
    if len(bear_divs) > 0:
        bearish_signals.append("⚠️ DIVERGENCIA BAJISTA detectada en el RSI")
        
    if last['Close'] <= last['BB_Lower']:
        bullish_signals.append("Precio toca Banda Inferior de Bollinger")
    elif last['Close'] >= last['BB_Upper']:
        bearish_signals.append("Precio toca Banda Superior de Bollinger")

    total_checks = len(bullish_signals) + len(bearish_signals)
    if total_checks == 0:
        return "NEUTRAL", 50, [], [], last['Close'], last['Close'], last['Close']

    bullish_score = (len(bullish_signals) / total_checks) * 100
    current_price = last['Close']
    
    if bullish_score >= 60:
        bias = "COMPRAR (BULLISH)"
        sl = last['Support'] * 0.995
        tp = current_price + (2 * (current_price - sl))
    elif bullish_score <= 40:
        bias = "VENTAR (BEARISH)"
        sl = last['Resistance'] * 1.005
        tp = current_price - (2 * (sl - current_price))
    else:
        bias = "NEUTRAL / ESPERAR"
        sl, tp = current_price, current_price
        
    return bias, bullish_score, bullish_signals, bearish_signals, current_price, sl, tp

# --- INTERFAZ STREAMLIT ---
st.title("📈 Pro Trading Confluence & Divergence Suite")
st.caption("Panel de análisis con detección de Divergencias RSI y Marcadores de Velas")

# Menú lateral
st.sidebar.header("Selección de Activo")
category = st.sidebar.selectbox("Categoría", list(ASSETS.keys()))
asset_name = st.sidebar.selectbox("Activo", list(ASSETS[category].keys()))
ticker = ASSETS[category][asset_name]

tf_selected = st.sidebar.selectbox("Temporalidad", list(TIMEFRAME_CONFIG.keys()), index=5)
tf_params = TIMEFRAME_CONFIG[tf_selected]

# Descargar datos
data = yf.download(ticker, period=tf_params["period"], interval=tf_params["interval"])

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    df = calculate_indicators(data)
    candle_annotations = detect_all_candlestick_patterns(df)
    bull_divs, bear_divs = detect_rsi_divergences(df)
    
    bias, score, bullish_reasons, bearish_reasons, entry, sl, tp = evaluate_confluence(df, bull_divs, bear_divs)
    
    # Métricas principales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"${entry:.2f}")
    c2.metric("Sesgo Dominante", bias)
    c3.metric("Confluencia Alcista", f"{score:.1f}%")
    c4.metric("Divergencias RSI", f"{len(bull_divs)} Alc. / {len(bear_divs)} Baj.")
    
    st.markdown("---")
    
    if "COMPRAR" in bias or "VENTA" in bias:
        st.subheader("🎯 Plan de Trading Recomendado")
        col_a, col_b, col_c = st.columns(3)
        col_a.info(f"**Precio de Entrada:** ${entry:.2f}")
        col_b.error(f"**Stop Loss:** ${sl:.2f}")
        col_c.success(f"**Take Profit:** ${tp:.2f}")

    # Pestañas para desglose
    t1, t2 = st.tabs(["📊 Gráfico de Velas e Indicadores", "💡 Desglose de Confluencias"])
    
    with t1:
        # Gráfico con dos paneles (Precio y RSI)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # 1. Velas Japonesas
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Precio"
        ), row=1, col=1)
        
        # Indicadores en Precio
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='cyan', width=1.5), name="SMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dash'), name="Bollinger Sup."), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dash'), name="Bollinger Inf."), row=1, col=1)
        
        # Dibujar marcas de patrones de velas
        for d_val, price_val, text_lbl, color_lbl, pos in candle_annotations:
            fig.add_annotation(
                x=d_val, y=price_val, text=text_lbl, showarrow=True,
                arrowhead=2, arrowcolor=color_lbl, arrowsize=1,
                font=dict(color=color_lbl, size=10), row=1, col=1
            )

        # Dibujar líneas de Divergencias en el gráfico de precio
        for div in bull_divs:
            fig.add_trace(go.Scatter(
                x=[div['p1_date'], div['p2_date']], y=[div['p1_price'], div['p2_price']],
                mode="lines+markers", line=dict(color="lime", width=3), name="Div. Alcista"
            ), row=1, col=1)
            
        for div in bear_divs:
            fig.add_trace(go.Scatter(
                x=[div['p1_date'], div['p2_date']], y=[div['p1_price'], div['p2_price']],
                mode="lines+markers", line=dict(color="crimson", width=3), name="Div. Bajista"
            ), row=1, col=1)
            
        # Niveles TP / SL
        if "COMPRAR" in bias or "VENTA" in bias:
            fig.add_hline(y=entry, line_dash="dash", line_color="blue", row=1, col=1)
            fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=tp, line_dash="dash", line_color="green", row=1, col=1)

        # 2. Subgráfico de RSI y sus Divergencias
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI (14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        for div in bull_divs:
            fig.add_trace(go.Scatter(
                x=[div['p1_date'], div['p2_date']], y=[div['p1_rsi'], div['p2_rsi']],
                mode="lines+markers", line=dict(color="lime", width=3), showlegend=False
            ), row=2, col=1)
            
        for div in bear_divs:
            fig.add_trace(go.Scatter(
                x=[div['p1_date'], div['p2_date']], y=[div['p1_rsi'], div['p2_rsi']],
                mode="lines+markers", line=dict(color="crimson", width=3), showlegend=False
            ), row=2, col=1)

        fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("🟢 **Factores Alcistas Confirmados:**")
            for reason in bullish_reasons:
                st.write(f"- {reason}")
        with col2:
            st.write("🔴 **Factores Bajistas Confirmados:**")
            for reason in bearish_reasons:
                st.write(f"- {reason}")
else:
    st.error("No se pudieron cargar datos para este activo en la temporalidad seleccionada.")
