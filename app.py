import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

st.set_page_config(page_title="Institutional Trading Confluence Engine", layout="wide", initial_sidebar_state="expanded")

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

TIMEFRAME_CONFIG = {
    "5 Minutos (5m)": {"interval": "5m", "period": "5d"},
    "15 Minutos (15m)": {"interval": "15m", "period": "1mo"},
    "30 Minutos (30m)": {"interval": "30m", "period": "1mo"},
    "1 Hora (1h)": {"interval": "1h", "period": "2mo"},
    "4 Horas (4h)": {"interval": "60m", "period": "3mo"},
    "1 Día (1d)": {"interval": "1d", "period": "1y"},
    "1 Semana (1wk)": {"interval": "1wk", "period": "3y"}
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
    df['BB_Mid'] = bb.bollinger_mavg()
    
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    return df

# --- DETECCIÓN DE PATRONES CHARTISTAS ---
def detect_chart_patterns(df, lookback=60):
    patterns = []
    df_sub = df.tail(lookback).copy()
    highs = df_sub['High'].values
    lows = df_sub['Low'].values
    closes = df_sub['Close'].values
    
    if len(df_sub) < 30:
        return patterns

    # HCH / HCH Invertido
    pivot_h = [i for i in range(5, len(highs)-5) if highs[i] == max(highs[i-5:i+6])]
    if len(pivot_h) >= 3:
        h1, h2, h3 = highs[pivot_h[-3]], highs[pivot_h[-2]], highs[pivot_h[-1]]
        if h2 > h1 and h2 > h3 and abs(h1 - h3) / h1 < 0.03:
            patterns.append(("Hombro-Cabeza-Hombro (HCH)", "Bearish", df_sub.index[pivot_h[-2]], h2))
            
    pivot_l = [i for i in range(5, len(lows)-5) if lows[i] == min(lows[i-5:i+6])]
    if len(pivot_l) >= 3:
        l1, l2, l3 = lows[pivot_l[-3]], lows[pivot_l[-2]], lows[pivot_l[-1]]
        if l2 < l1 and l2 < l3 and abs(l1 - l3) / l1 < 0.03:
            patterns.append(("HCH Invertido", "Bullish", df_sub.index[pivot_l[-2]], l2))

    # Triángulos y Rectángulos
    max_recent = max(highs[-20:])
    min_recent = min(lows[-20:])
    range_pct = (max_recent - min_recent) / min_recent if min_recent != 0 else 0
    
    x = np.arange(15)
    slope_highs = np.polyfit(x, highs[-15:], 1)[0]
    slope_lows = np.polyfit(x, lows[-15:], 1)[0]
    
    if abs(slope_highs) < 0.001 and slope_lows > 0.01:
        patterns.append(("Triángulo Ascendente", "Bullish", df_sub.index[-1], max_recent))
    elif slope_highs < -0.01 and abs(slope_lows) < 0.001:
        patterns.append(("Triángulo Descendente", "Bearish", df_sub.index[-1], min_recent))
    elif slope_highs < -0.005 and slope_lows > 0.005:
        patterns.append(("Triángulo Simétrico", "Neutral", df_sub.index[-1], closes[-1]))
    elif range_pct < 0.03:
        patterns.append(("Rectángulo de Continuación", "Neutral", df_sub.index[-1], closes[-1]))
        
    # Banderas
    impulse = (closes[-1] - closes[-25]) / closes[-25] if closes[-25] != 0 else 0
    if impulse > 0.05 and range_pct < 0.025:
        patterns.append(("Bandera Alcista", "Bullish", df_sub.index[-1], closes[-1]))
    elif impulse < -0.05 and range_pct < 0.025:
        patterns.append(("Bandera Bajista", "Bearish", df_sub.index[-1], closes[-1]))
        
    return patterns

# --- DETECCIÓN DE PATRONES DE VELAS ---
def detect_candlestick_patterns(df, lookback=40):
    annotations = []
    start_idx = max(2, len(df) - lookback)
    
    for i in range(start_idx, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        body = abs(curr['Close'] - curr['Open'])
        range_c = curr['High'] - curr['Low']
        if range_c == 0:
            continue
        
        lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
        upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
        d_val = df.index[i]
        
        # Alcistas
        if lower_wick > 2 * body and upper_wick < body:
            annotations.append((d_val, curr['Low'], "Martillo", "green", "bottom"))
        elif upper_wick > 2 * body and lower_wick < body:
            annotations.append((d_val, curr['High'], "Martillo Invertido", "green", "bottom"))
        elif curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and curr['Close'] >= prev['Open'] and curr['Open'] <= prev['Close']:
            annotations.append((d_val, curr['Low'], "Env. Alcista", "green", "bottom"))
        elif prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and curr['Open'] > prev['Low'] and curr['Close'] > ((prev['Open'] + prev['Close']) / 2):
            annotations.append((d_val, curr['Low'], "Pauta Penetrante", "green", "bottom"))
            
        # Bajistas
        elif upper_wick > 2 * body and lower_wick < body:
            annotations.append((d_val, curr['High'], "Estrella Fugaz", "red", "top"))
        elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and curr['Close'] <= prev['Open'] and curr['Open'] >= prev['Close']:
            annotations.append((d_val, curr['High'], "Env. Bajista", "red", "top"))
        elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and curr['Open'] > prev['High'] and curr['Close'] < ((prev['Open'] + prev['Close']) / 2):
            annotations.append((d_val, curr['High'], "Cubierta Nube Oscura", "red", "top"))
            
    return annotations

# --- DIVERGENCIAS RSI ---
def detect_rsi_divergences(df, window=5, lookback=60):
    df_sub = df.tail(lookback).copy()
    bullish_divs, bearish_divs = [], []
    lows = df_sub['Low'].values
    highs = df_sub['High'].values
    rsi = df_sub['RSI'].values
    dates = df_sub.index
    
    pivot_l = [i for i in range(window, len(df_sub)-window) if lows[i] == min(lows[i-window:i+window+1])]
    pivot_h = [i for i in range(window, len(df_sub)-window) if highs[i] == max(highs[i-window:i+window+1])]
    
    for k in range(1, len(pivot_l)):
        i1, i2 = pivot_l[k-1], pivot_l[k]
        if lows[i2] < lows[i1] and rsi[i2] > rsi[i1]:
            bullish_divs.append({'p1_date': dates[i1], 'p1_price': lows[i1], 'p1_rsi': rsi[i1], 'p2_date': dates[i2], 'p2_price': lows[i2], 'p2_rsi': rsi[i2]})
            
    for k in range(1, len(pivot_h)):
        i1, i2 = pivot_h[k-1], pivot_h[k]
        if highs[i2] > highs[i1] and rsi[i2] < rsi[i1]:
            bearish_divs.append({'p1_date': dates[i1], 'p1_price': highs[i1], 'p1_rsi': rsi[i1], 'p2_date': dates[i2], 'p2_price': highs[i2], 'p2_rsi': rsi[i2]})
            
    return bullish_divs, bearish_divs

# --- MOTOR DE CONFLUENCIA PONDERADO ---
def evaluate_high_probability_confluence(df, chart_patterns, candle_annotations, bull_divs, bear_divs):
    last = df.iloc[-1]
    score_bull, score_bear = 0, 0
    bullish_reasons, bearish_reasons = [], []

    # 1. Alineación de Tendencia y Medias Móviles
    if last['Close'] > last['SMA_200']:
        score_bull += 15
        bullish_reasons.append("Estructura sobre SMA 200 (Tendencia Principal Alcista)")
    else:
        score_bear += 15
        bearish_reasons.append("Estructura bajo SMA 200 (Tendencia Principal Bajista)")

    if last['EMA_20'] > last['EMA_50']:
        score_bull += 10
        bullish_reasons.append("Cruce de Medias Rápido (EMA 20 > EMA 50)")
    else:
        score_bear += 10
        bearish_reasons.append("Cruce de Medias Bajista (EMA 20 < EMA 50)")

    # 2. Divergencias y Osciladores RSI/Estocástico
    if len(bull_divs) > 0:
        score_bull += 15
        bullish_reasons.append("Divergencia Alcista RSI Confirmada")
    if len(bear_divs) > 0:
        score_bear += 15
        bearish_reasons.append("Divergencia Bajista RSI Confirmada")

    if last['RSI'] < 35:
        score_bull += 5
        bullish_reasons.append(f"RSI en Zona de Sobreventa ({last['RSI']:.1f})")
    elif last['RSI'] > 65:
        score_bear += 5
        bearish_reasons.append(f"RSI en Zona de Sobrecompra ({last['RSI']:.1f})")

    if last['Stoch_K'] < 20 and last['Stoch_K'] > last['Stoch_D']:
        score_bull += 5
        bullish_reasons.append("Cruce Alcista de Estocástico en Sobreventa")
    elif last['Stoch_K'] > 80 and last['Stoch_K'] < last['Stoch_D']:
        score_bear += 5
        bearish_reasons.append("Cruce Bajista de Estocástico en Sobrecompra")

    # 3. Patrones Chartistas
    for p_name, p_type, _, _ in chart_patterns:
        if p_type == "Bullish":
            score_bull += 25
            bullish_reasons.append(f"Patrón Chartista Alcista: {p_name}")
        elif p_type == "Bearish":
            score_bear += 25
            bearish_reasons.append(f"Patrón Chartista Bajista: {p_name}")

    # 4. Volatilidad y Volumen Institucional
    if last['Close'] <= last['BB_Lower']:
        score_bull += 10
        bullish_reasons.append("Precio en la Banda Inferior de Bollinger")
    elif last['Close'] >= last['BB_Upper']:
        score_bear += 10
        bearish_reasons.append("Precio en la Banda Superior de Bollinger")

    if last['Volume'] > 1.3 * last['Vol_SMA']:
        if last['Close'] > last['Open']:
            score_bull += 5
            bullish_reasons.append("Volumen Institucional superior a la media")
        else:
            score_bear += 5
            bearish_reasons.append("Volumen Institucional superior a la media")

    # 5. Patrones de Velas Japonesas
    recent_candles = candle_annotations[-3:] if len(candle_annotations) >= 3 else candle_annotations
    for _, _, label, col, _ in recent_candles:
        if col == "green":
            score_bull += 5
            bullish_reasons.append(f"Vela Alcista Reciente: {label}")
        elif col == "red":
            score_bear += 5
            bearish_reasons.append(f"Vela Bajista Reciente: {label}")

    # Cálculo final y Gestión de Riesgo (ATR)
    total_score = max(score_bull, score_bear)
    current_price = last['Close']
    atr_val = last['ATR'] if not np.isnan(last['ATR']) else current_price * 0.015

    if score_bull >= 65 and score_bull > score_bear:
        bias = "COMPRAR (ALTA CONFLUENCIA ALCISTA)"
        sl = current_price - (1.5 * atr_val)
        tp = current_price + (3.0 * atr_val)
        confluence_pct = score_bull
    elif score_bear >= 65 and score_bear > score_bull:
        bias = "VENTAR (ALTA CONFLUENCIA BAJISTA)"
        sl = current_price + (1.5 * atr_val)
        tp = current_price - (3.0 * atr_val)
        confluence_pct = score_bear
    else:
        bias = "NEUTRAL / NO OPERAR (CONFLUENCIA INSUFICIENTE)"
        sl, tp = current_price, current_price
        confluence_pct = total_score

    return bias, min(confluence_pct, 100), bullish_reasons, bearish_reasons, current_price, sl, tp

# --- INTERFAZ STREAMLIT ---
st.title("🛡️ Institutional Confluence & Technical Analysis Suite")
st.caption("Motor cuantitativo con validación de patrones chartistas, volumen e indicadores combinados")

# Menú lateral
st.sidebar.header("Parámetros del Activo")
category = st.sidebar.selectbox("Categoría", list(ASSETS.keys()))
asset_name = st.sidebar.selectbox("Activo", list(ASSETS[category].keys()))
ticker = ASSETS[category][asset_name]

tf_selected = st.sidebar.selectbox("Temporalidad", list(TIMEFRAME_CONFIG.keys()), index=5)
tf_params = TIMEFRAME_CONFIG[tf_selected]

# Descarga de datos
data = yf.download(ticker, period=tf_params["period"], interval=tf_params["interval"])

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    df = calculate_indicators(data)
    chart_patterns = detect_chart_patterns(df)
    candle_annotations = detect_candlestick_patterns(df)
    bull_divs, bear_divs = detect_rsi_divergences(df)
    
    bias, score, bullish_reasons, bearish_reasons, entry, sl, tp = evaluate_high_probability_confluence(
        df, chart_patterns, candle_annotations, bull_divs, bear_divs
    )
    
    # Métricas Principales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"${entry:.2f}")
    c2.metric("Sesgo de Operación", bias)
    c3.metric("Puntuación de Confluencia", f"{score:.0f}%")
    c4.metric("Patrones Detectados", f"{len(chart_patterns)} Chart. / {len(candle_annotations)} Velas")
    
    st.markdown("---")
    
    if "COMPRAR" in bias or "VENTA" in bias:
        st.subheader("🎯 Plan Operativo de Alta Probabilidad")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.info(f"**Precio Entrada:** ${entry:.2f}")
        col_b.error(f"**Stop Loss (ATR):** ${sl:.2f}")
        col_c.success(f"**Take Profit (R:R 1:2):** ${tp:.2f}")
        col_d.metric("Ratio Riesgo:Beneficio", "1 : 2.0")

    tabs = st.tabs(["📊 Gráfico de Análisis Técnico", "🔍 Confluencias Confirmadas", "📚 Normas de Estrategias y Conceptos"])
    
    with tabs[0]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # Velas Japonesas
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
        
        # Medias e Indicadores
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='cyan', width=1.5), name="SMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dash'), name="BB Sup."), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dash'), name="BB Inf."), row=1, col=1)
        
        # Anotaciones de Velas
        for d_val, price_val, text_lbl, color_lbl, _ in candle_annotations[-10:]:
            fig.add_annotation(x=d_val, y=price_val, text=text_lbl, showarrow=True, arrowhead=2, arrowcolor=color_lbl, font=dict(color=color_lbl, size=9), row=1, col=1)

        # Anotaciones de Patrones Chartistas
        for p_name, p_type, p_date, p_price in chart_patterns:
            p_color = "lime" if p_type == "Bullish" else "red" if p_type == "Bearish" else "yellow"
            fig.add_annotation(x=p_date, y=p_price, text=f"📐 {p_name}", showarrow=True, arrowhead=4, arrowcolor=p_color, font=dict(color=p_color, size=11), row=1, col=1)

        # Divergencias RSI en Gráfico de Precio
        for div in bull_divs:
            fig.add_trace(go.Scatter(x=[div['p1_date'], div['p2_date']], y=[div['p1_price'], div['p2_price']], mode="lines+markers", line=dict(color="lime", width=3), name="Div. Alcista"), row=1, col=1)
        for div in bear_divs:
            fig.add_trace(go.Scatter(x=[div['p1_date'], div['p2_date']], y=[div['p1_price'], div['p2_price']], mode="lines+markers", line=dict(color="crimson", width=3), name="Div. Bajista"), row=1, col=1)

        # Niveles TP / SL
        if "COMPRAR" in bias or "VENTA" in bias:
            fig.add_hline(y=entry, line_dash="dash", line_color="blue", row=1, col=1)
            fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=tp, line_dash="dash", line_color="green", row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI (14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("🟢 **Factores Alcistas Detectados:**")
            for r in bullish_reasons:
                st.write(f"- {r}")
        with col2:
            st.write("🔴 **Factores Bajistas Detectados:**")
            for r in bearish_reasons:
                st.write(f"- {r}")

    with tabs[2]:
        st.markdown("""
        ### 📖 Reglas de Estrategias e Indicadores Implementados

        #### 1. Estrategia de Confluencia de Tendencia (SMA 200 + EMA 20/50)
        * **Regla Alcista:** El precio debe situarse por encima de la SMA 200 y la EMA 20 debe estar por encima de la EMA 50.
        * **Regla Bajista:** El precio se sitúa por debajo de la SMA 200 y la EMA 20 está por debajo de la EMA 50.

        #### 2. Reversión por Divergencia de RSI + Bandas de Bollinger
        * **Divergencia Alcista:** Se forma cuando el precio marca un mínimo más bajo pero el indicador RSI marca un mínimo más alto.
        * **Divergencia Bajista:** Ocurre cuando el precio marca un máximo más alto pero el RSI marca un máximo más bajo.

        #### 3. Validación por Patrones Chartistas
        * **Hombro-Cabeza-Hombro (HCH):** Patrón de cambio de tendencia bajista.
        * **Triángulos (Ascendente/Descendente/Simétrico):** Patrones de consolidación y ruptura.
        * **Banderas Alcistas/Bajistas:** Patrones de continuación de tendencia.

        #### 4. Gestión de Riesgo ATR (Average True Range)
        * **Stop Loss:** Se calcula a 1.5 veces el valor del ATR desde el precio de entrada.
        * **Take Profit:** Ratio mínimo de Beneficio/Riesgo de **2:1** (3.0 veces el ATR).
        """)
else:
    st.error("No se pudieron cargar datos para este activo.")
