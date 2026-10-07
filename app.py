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

# --- DETECCIÓN DE PATRONES CHARTISTAS (ESTRUCTURAS) ---
def detect_chart_patterns(df, lookback=60):
    patterns = []
    df_sub = df.tail(lookback).copy()
    highs = df_sub['High'].values
    lows = df_sub['Low'].values
    closes = df_sub['Close'].values
    
    if len(df_sub) < 30:
        return patterns

    # 1. Hombro-Cabeza-Hombro (HCH) / HCH Invertido
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

    # 2. Triángulos y Rectángulos
    max_recent = max(highs[-20:])
    min_recent = min(lows[-20:])
    range_pct = (max_recent - min_recent) / min_recent
    
    # Pendientes para Triángulos
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
        
    # 3. Banderas (Consolidación tras impulso fuerte)
    impulse = (closes[-1] - closes[-25]) / closes[-25]
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
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        body = abs(curr['Close'] - curr['Open'])
        range_c = curr['High'] - curr['Low']
        if range_c == 0: continue
        
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
        elif prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and curr['Open'] > prev['Low'] and curr['Close'] > (prev['Open'] + prev['Close'])/2:
            annotations.append((d_val, curr['Low'], "Pauta Penetrante", "green", "bottom"))
            
        # Bajistas
        elif upper_wick > 2 * body and lower_wick < body:
            annotations.append((d_val, curr['High'], "Estrella Fugaz", "red", "top"))
        elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and curr['Close'] <= prev['Open'] and curr['Open'] >= prev['Close']:
            annotations.append((d_val
