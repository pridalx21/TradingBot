import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
import yfinance as yf
import requests
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
import os
from dotenv import load_dotenv
import altair as alt
from trading_bot import MultiAssetTradingBot

# Funktion zum Abrufen von Marktdaten
def fetch_market_data(symbol='BTC/USDT', timeframe='1h', limit=100):
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
        'timeout': 60000,  # Timeout auf 60 Sekunden erhöhen
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Debugging: Log the fetched market data
    st.write("Marktdaten:", df.head())
    
    return df

# Bitpanda API-Integration
load_dotenv()
api_key = os.getenv("BITPANDA_API_KEY")

if api_key:
    st.success("API-Schlüssel erfolgreich geladen.")
else:
    st.error("Kein gültiger API-Schlüssel verfügbar.")

# Titel und Beschreibung
st.title("Krypto Marktanalyse und Daytrading Tipps")
st.write("""
Diese Anwendung bietet Echtzeit-Marktdaten, technische Indikatoren und Handelssignale für verschiedene Kryptowährungen.
""")

# Seitenleiste für Benutzereingaben
st.sidebar.header("Einstellungen")

# Automatische Auswahl und gleichzeitiges Trading mit mehreren Handelspaaren
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]
selected_symbols = []
performances = {}

for symbol in symbols:
    data = fetch_market_data(symbol, timeframe="1h", limit=100)
    performance = data['close'].pct_change().sum()
    performances[symbol] = performance

# Wähle die Top 3 Handelspaare basierend auf der Performance
selected_symbols = sorted(performances, key=performances.get, reverse=True)[:3]

st.write(f"Automatisch ausgewählte Handelspaare: {', '.join(selected_symbols)}")

# Auswahl des Zeitrahmens
timeframe = st.sidebar.selectbox(
    "Wähle den Zeitrahmen",
    ("1m", "5m", "15m", "1h", "4h", "1d")
)

# Anzahl der Datenpunkte
limit = st.sidebar.number_input("Anzahl der Datenpunkte", min_value=50, max_value=1000, value=100)

# Menü für Bitpanda-Datenübersicht hinzufügen
if st.sidebar.button("Bitpanda Datenübersicht"):
    if api_key:
        # Beispielaufruf der Bitpanda API
        headers = {
            'X-API-KEY': api_key,
            'Accept': 'application/json',
        }
        response = requests.get("https://api.bitpanda.com/v1/wallets", headers=headers)
        if response.status_code == 200:
            wallets = response.json()
            
            # Entferne Debugging-Ausgabe
            # st.write("### Debugging: Bitpanda API Response")
            # st.json(response.json())

            # Bitpanda Wallets Übersicht
            st.write("### Bitpanda Wallets Übersicht")
            for wallet in wallets['data']:
                attributes = wallet['attributes']
                st.write(f"**{attributes.get('name', 'Unknown')}**: {attributes.get('balance', '0')} {attributes.get('cryptocoin_symbol', 'Unknown')}")
        else:
            st.error("Fehler beim Abrufen der Bitpanda-Daten.")
    else:
        st.error("API-Schlüssel nicht gefunden. Bitte überprüfen Sie Ihre .env-Datei.")

# BTC Wallet Balance abrufen
btc_wallet_balance = 0  # Hier den tatsächlichen Kontostand abrufen

# Beispiel: Abrufen des BTC Wallet Balance von Bitpanda
try:
    response = requests.get('https://api.bitpanda.com/v1/wallets', headers={'X-Api-Key': api_key})
    if response.status_code == 200:
        wallets = response.json().get('data', [])
        for wallet in wallets:
            attributes = wallet.get('attributes', {})
            if attributes.get('currency') == 'BTC':
                btc_wallet_balance = float(attributes.get('balance', 0))
                break
    else:
        st.error("Fehler beim Abrufen des BTC Wallet Balance")
except Exception as e:
    st.error(f"Fehler: {str(e)}")

account_size = btc_wallet_balance

# Risikoprozentsatz definieren
risk_percentage = 5

# Trading Bot Integration
st.sidebar.markdown("---")
st.sidebar.header("🤖 Trading Bot")

# Bot Konfiguration
bot_enabled = st.sidebar.checkbox("Trading Bot aktivieren", value=False)
if bot_enabled:
    bot_timeframe = st.sidebar.selectbox(
        "Trading Intervall",
        ("1m", "5m", "15m", "1h"),
        index=0
    )
    
    # Asset-Auswahl
    st.sidebar.subheader("Asset-Klassen")
    trade_crypto = st.sidebar.checkbox("Kryptowährungen", value=True)
    trade_stocks = st.sidebar.checkbox("Aktien", value=False)
    trade_etfs = st.sidebar.checkbox("ETFs", value=False)
    trade_metals = st.sidebar.checkbox("Edelmetalle", value=False)
    
    # Leverage-Einstellungen
    st.sidebar.subheader("Hebel-Einstellungen")
    leverage_crypto = st.sidebar.number_input("Krypto Hebel", min_value=1, max_value=5, value=3)
    leverage_stocks = st.sidebar.number_input("Aktien Hebel", min_value=1, max_value=3, value=2)
    leverage_etfs = st.sidebar.number_input("ETF Hebel", min_value=1, max_value=2, value=1)
    leverage_metals = st.sidebar.number_input("Metall Hebel", min_value=1, max_value=4, value=2)
    
    # Stop-Loss/Take-Profit
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.number_input("Stop-Loss (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    take_profit = st.sidebar.number_input("Take-Profit (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

# Trading Bot Status und Performance
if bot_enabled:
    st.markdown("## 🤖 Trading Bot Status")
    
    # Erstelle Spalten für das Layout
    col1, col2, col3 = st.columns(3)
    
    # Bot-Status
    with col1:
        st.metric(
            "Bot Status",
            "Aktiv ✅",
            "Trading läuft"
        )
    
    # Gesamtperformance
    with col2:
        st.metric(
            "Trades Heute",
            "0",
            "Warte auf Signale..."
        )
    
    with col3:
        st.metric(
            "Gewinn/Verlust",
            "€0.00",
            "0%"
        )
    
    # Performance pro Asset-Klasse
    st.markdown("### 📊 Performance nach Asset-Klasse")
    
    # Erstelle Tabs für verschiedene Asset-Klassen
    asset_tabs = st.tabs(["Krypto", "Aktien", "ETFs", "Metalle"])
    
    for tab, (asset_type, enabled) in zip(
        asset_tabs,
        [
            ("crypto", trade_crypto),
            ("stocks", trade_stocks),
            ("etfs", trade_etfs),
            ("metals", trade_metals)
        ]
    ):
        with tab:
            if enabled:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Offene Positionen",
                        "0",
                        "Keine aktiven Trades"
                    )
                    
                    st.metric(
                        "Trades Heute",
                        "0",
                        "Warte auf Signale..."
                    )
                
                with col2:
                    st.metric(
                        "Gewinn/Verlust",
                        "€0.00",
                        "0%"
                    )
                    
                    st.metric(
                        "Bester Trade",
                        "€0.00",
                        "0%"
                    )
            else:
                st.info(f"{asset_type.upper()} Trading deaktiviert")
    
    # Live Trading Feed
    st.markdown("### 📈 Live Trading Feed")
    
    # Erstelle einen leeren Container für Updates
    trading_feed = st.empty()
    
    # Platzhalter für Trading Feed
    with trading_feed.container():
        st.info("Trading Bot initialisiert. Warte auf Handelssignale...")
    
    # Performance Charts
    st.markdown("### 📊 Performance Übersicht")
    
    # Initialisiere Beispieldaten für Charts
    now = pd.Timestamp.now()
    date_range = pd.date_range(start=now - pd.Timedelta(days=30), end=now, freq='H')
    
    # Generiere validierte Testdaten
    np.random.seed(42)  # Für reproduzierbare Ergebnisse
    profits = np.random.randn(len(date_range)).cumsum()
    
    chart_data = pd.DataFrame({
        'timestamp': date_range,
        'profit': profits,
    })
    
    # Stelle sicher, dass keine unendlichen oder NaN-Werte vorhanden sind
    chart_data = chart_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Konvertiere Timestamp zu Datetime und stelle sicher, dass es gültige Werte sind
    chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
    chart_data = chart_data[chart_data['timestamp'].notna()]
    
    # Sortiere nach Timestamp
    chart_data = chart_data.sort_values('timestamp')
    
    def create_performance_chart(data, height=300):
        if data.empty:
            return alt.Chart().mark_text().encode(
                text=alt.value('Keine Daten verfügbar')
            )

        # Berechne y-Achsen Grenzen mit Puffer
        y_min = data['profit'].min()
        y_max = data['profit'].max()
        y_padding = (y_max - y_min) * 0.1 if y_min != y_max else 1.0  # Verhindere Division durch Null
        
        # Berechne x-Achsen Grenzen
        x_min = data['timestamp'].min()
        x_max = data['timestamp'].max()
        
        if pd.isna(x_min) or pd.isna(x_max):
            return alt.Chart().mark_text().encode(
                text=alt.value('Ungültige Zeitstempel in den Daten')
            )
        
        base = alt.Chart(data).encode(
            x=alt.X('timestamp:T', 
                   title='Zeit',
                   axis=alt.Axis(labelAngle=-45),
                   scale=alt.Scale(domain=[x_min, x_max], nice=True)
            )
        ).properties(height=height)

        line = base.mark_line(color='#00ff00').encode(
            y=alt.Y('profit:Q',
                   title='Gewinn/Verlust (EUR)',
                   scale=alt.Scale(
                       domain=[y_min - y_padding, y_max + y_padding],
                       nice=True
                   ),
                   axis=alt.Axis(format='.2f')
            ),
            tooltip=[
                alt.Tooltip('timestamp:T', title='Zeit', format='%Y-%m-%d %H:%M'),
                alt.Tooltip('profit:Q', title='Gewinn/Verlust', format='.2f')
            ]
        )

        return line

    # Erstelle Tabs für verschiedene Zeiträume
    time_tabs = st.tabs(["24h", "7d", "30d"])
    
    with time_tabs[0]:  # 24h
        cutoff_24h = now - pd.Timedelta(hours=24)
        last_24h = chart_data[chart_data['timestamp'] >= cutoff_24h].copy()
        if not last_24h.empty:
            st.altair_chart(
                create_performance_chart(last_24h),
                use_container_width=True
            )
        else:
            st.warning("Keine Daten für die letzten 24 Stunden verfügbar")
    
    with time_tabs[1]:  # 7d
        cutoff_7d = now - pd.Timedelta(days=7)
        last_7d = chart_data[chart_data['timestamp'] >= cutoff_7d].copy()
        if not last_7d.empty:
            st.altair_chart(
                create_performance_chart(last_7d),
                use_container_width=True
            )
        else:
            st.warning("Keine Daten für die letzten 7 Tage verfügbar")
    
    with time_tabs[2]:  # 30d
        if not chart_data.empty:
            st.altair_chart(
                create_performance_chart(chart_data),
                use_container_width=True
            )
        else:
            st.warning("Keine Daten für die letzten 30 Tage verfügbar")
    
    # Risk Management Übersicht
    st.markdown("### ⚠️ Risk Management")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        # Aktive Positionen
        st.markdown("#### Aktive Positionen")
        positions_df = pd.DataFrame({
            'Asset': ['BTC/EUR', 'ETH/EUR', 'XRP/EUR'],
            'Typ': ['Long', 'Short', 'Long'],
            'Einstieg': ['€41,000', '€2,200', '€0.55'],
            'Stop-Loss': ['€40,590', '€2,222', '€0.54'],
            'Take-Profit': ['€41,820', '€2,156', '€0.56']
        })
        st.dataframe(
            positions_df.style.apply(lambda x: ['background-color: #1e2130'] * len(x)),
            hide_index=True,
            use_container_width=True
        )
    
    with risk_col2:
        # Risk Exposure
        st.markdown("#### Risk Exposure")
        exposure_df = pd.DataFrame({
            'Asset-Klasse': ['Krypto', 'Aktien', 'ETFs', 'Metalle'],
            'Risk %': [2.5, 1.2, 0.8, 0.5]
        })
        
        # Erstelle Balkendiagramm für Risk Exposure
        risk_chart = alt.Chart(exposure_df).mark_bar().encode(
            x=alt.X('Asset-Klasse:N', title='Asset-Klasse'),
            y=alt.Y('Risk %:Q', 
                    title='Risiko (%)', 
                    scale=alt.Scale(domain=[0, 5])),
            color=alt.Color(
                'Asset-Klasse:N',
                legend=None,
                scale=alt.Scale(
                    domain=['Krypto', 'Aktien', 'ETFs', 'Metalle'],
                    range=['#00ff00', '#ff9900', '#00ccff', '#ff3366']
                )
            ),
            tooltip=[
                alt.Tooltip('Asset-Klasse:N', title='Asset'),
                alt.Tooltip('Risk %:Q', title='Risiko', format='.1f')
            ]
        ).properties(height=200)
        
        st.altair_chart(risk_chart, use_container_width=True)

# Starte den Trading Bot
if bot_enabled:
    # Konfiguriere Asset-Parameter
    asset_params = {
        'crypto': {
            'leverage': leverage_crypto,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'enabled': trade_crypto
        },
        'stocks': {
            'leverage': leverage_stocks,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'enabled': trade_stocks
        },
        'etfs': {
            'leverage': leverage_etfs,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'enabled': trade_etfs
        },
        'metals': {
            'leverage': leverage_metals,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'enabled': trade_metals
        }
    }
    
    # Starte den Bot direkt im Streamlit-Kontext
    try:
        with st.spinner('Trading Bot wird gestartet...'):
            bot = MultiAssetTradingBot(
                timeframe=bot_timeframe,
                risk_percentage=risk_percentage
            )
            bot.asset_params = asset_params
            
            # Erstelle einen Container für Bot-Updates
            update_container = st.empty()
            
            def update_status(message):
                with update_container:
                    st.write(message)
            
            # Starte Trading mit Status-Updates
            bot.start_trading(status_callback=update_status)
            st.success("Trading Bot läuft!")
            
    except Exception as e:
        st.error(f"Bot Error: {str(e)}")

# Funktion zum Berechnen des gleitenden Durchschnitts
def moving_average(data, window):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    return data['close'].rolling(window=window).mean()

# Funktion zur Generierung von Handelssignalen
def generate_signal(data):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    data['MA_20'] = moving_average(data, 20)
    data['MA_50'] = moving_average(data, 50)
    signal = []
    for i in range(len(data)):
        if data['MA_20'][i] > data['MA_50'][i]:
            signal.append('Kaufen')
        elif data['MA_20'][i] < data['MA_50'][i]:
            signal.append('Verkaufen')
        else:
            signal.append('Halten')
    data['Signal'] = signal
    
    # Debugging: Log the generated signals
    st.write("Generierte Signale:", data[['timestamp', 'Signal']].tail())
    st.write("MA_20:", data['MA_20'].tail())
    st.write("MA_50:", data['MA_50'].tail())
    
    return data

# Bollinger Bands Berechnung
def calculate_bollinger_bands(data, window=20, num_std=2):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

# MACD Berechnung
def calculate_macd(data, fast=12, slow=26, signal=9):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Volatilitätsberechnung
def calculate_volatility(data, window=14):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    return data['close'].pct_change().rolling(window=window).std() * np.sqrt(window)

# Relative Strength Index (RSI)
def calculate_rsi(prices, window=14):
    # Berechne RSI direkt aus der Preisserie
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Trendstärke berechnen
def calculate_trend_strength(data, window=14):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    price_changes = data['close'].diff()
    positive_moves = price_changes.where(price_changes > 0, 0).rolling(window=window).mean()
    negative_moves = abs(price_changes.where(price_changes < 0, 0)).rolling(window=window).mean()
    trend_strength = (positive_moves / (positive_moves + negative_moves) * 100)
    return trend_strength

# Support und Resistance Levels finden
def find_support_resistance(data, window=20):
    if 'high' not in data.columns or 'low' not in data.columns:
        st.error("Fehler: 'high' oder 'low'-Spalte fehlt in den Daten")
        return None
    highs = data['high'].rolling(window=window, center=True).max()
    lows = data['low'].rolling(window=window, center=True).min()
    return lows.iloc[-1], highs.iloc[-1]

# Risikokalkulator
def calculate_position_size(account_size, risk_percentage, entry_price, stop_loss):
    risk_amount = account_size * (risk_percentage / 100)
    price_difference = abs(entry_price - stop_loss)
    position_size = risk_amount / price_difference
    return position_size

# Marktanalyse und Empfehlungen
def generate_market_analysis(data):
    if 'close' not in data.columns:
        st.error("Fehler: 'close'-Spalte fehlt in den Daten")
        return None
    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100

    volatility = calculate_volatility(data).iloc[-1] * 100
    rsi_value = data['RSI'].iloc[-1]

    analysis = []

    if rsi_value > 70:
        analysis.append("⚠️ **Überkauft**: Der RSI ist hoch ({}), was auf eine mögliche Korrektur hinweisen könnte.".format(round(rsi_value, 2)))
    elif rsi_value < 30:
        analysis.append("💡 **Überverkauft**: Der RSI ist niedrig ({}), was auf eine mögliche Erholung hinweisen könnte.".format(round(rsi_value, 2)))

    if volatility > 5:
        analysis.append("📊 **Hohe Volatilität**: {}% - Vorsicht bei Trades, enge Stop-Loss setzen.".format(round(volatility, 2)))
    else:
        analysis.append("📊 **Moderate Volatilität**: {}% - Normalere Marktbedingungen.".format(round(volatility, 2)))

    if abs(price_change) > 5:
        analysis.append("🔄 **Starke Preisbewegung**: {:.2f}% in der letzten Periode".format(price_change))

    return analysis

# Weltmarkt-Daten abrufen
def fetch_global_market_data():
    # Wichtige Markt-Indikatoren
    tickers = {
        'S&P 500': '^GSPC',
        'Gold': 'GC=F',
        'US Dollar Index': 'DX-Y.NYB',
        'Volatilitätsindex': '^VIX',
        'US 10Y Treasury': '^TNX'
    }
    
    market_data = {}
    for name, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(period='1d')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                daily_change = ((current_price - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100
                market_data[name] = {
                    'price': current_price,
                    'change': daily_change
                }
        except Exception as e:
            market_data[name] = {'price': None, 'change': None}
    
    return market_data

# Fear & Greed Index abrufen
def fetch_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        data = response.json()
        return {
            'value': data['data'][0]['value'],
            'classification': data['data'][0]['value_classification']
        }
    except:
        return {'value': '?', 'classification': 'Keine Daten verfügbar'}

# Nachrichtenabruf und Analyse
def fetch_crypto_news():
    try:
        newsapi = NewsApiClient(api_key='eba5f86a4e16496fbaf275549bfb9d33')  # Hier API-Key einfügen
        
        # Krypto-bezogene Nachrichten
        crypto_news = newsapi.get_everything(
            q='(bitcoin OR cryptocurrency OR crypto) AND (market OR price OR regulation)',
            language='de',
            sort_by='publishedAt',
            page_size=5
        )
        
        # Wirtschaftsnachrichten
        economic_news = newsapi.get_everything(
            q='(inflation OR "federal reserve" OR "central bank" OR recession)',
            language='de',
            sort_by='publishedAt',
            page_size=5
        )
        
        return {
            'crypto': crypto_news.get('articles', []),
            'economic': economic_news.get('articles', [])
        }
    except:
        return {'crypto': [], 'economic': []}

# Sentiment-Analyse für Nachrichten
def analyze_news_sentiment(text):
    try:
        analysis = TextBlob(text)
        # Normalisiere Sentiment auf Skala von -100 bis 100
        sentiment = analysis.sentiment.polarity * 100
        
        if sentiment > 30:
            return "Positiv", sentiment
        elif sentiment < -30:
            return "Negativ", sentiment
        else:
            return "Neutral", sentiment
    except:
        return "Neutral", 0

# Nachrichteneinfluss analysieren
def analyze_market_impact(news_item):
    title = news_item.get('title', '') or ''
    description = news_item.get('description', '') or ''
    
    # Wichtige Marktkatalysatoren
    keywords = {
        'sehr negativ': ['verbot', 'crash', 'hack', 'betrug', 'konkurs'],
        'negativ': ['regulierung', 'warnung', 'risiko', 'inflation'],
        'positiv': ['adoption', 'integration', 'partnerschaft', 'innovation'],
        'sehr positiv': ['durchbruch', 'genehmigung', 'mainstream', 'institutional']
    }
    
    combined_text = (title + ' ' + description).lower()
    impact = 0
    
    for sentiment, words in keywords.items():
        for word in words:
            if word in combined_text:
                if sentiment == 'sehr negativ':
                    impact -= 2
                elif sentiment == 'negativ':
                    impact -= 1
                elif sentiment == 'positiv':
                    impact += 1
                else:  # sehr positiv
                    impact += 2
    
    return impact

# Daten abrufen und analysieren
with st.spinner('Daten werden abgerufen und analysiert...'):
    for symbol in selected_symbols:
        data = fetch_market_data(symbol, timeframe, limit)
        
        # Sicherstellen, dass die 'close'-Spalte vorhanden ist
        if 'close' not in data.columns:
            st.error(f"Fehler: 'close'-Spalte fehlt in den Daten für {symbol}")
            continue
        
        data = generate_signal(data)
        data['RSI'] = calculate_rsi(data['close'])

        last_signal = data['Signal'].iloc[-1]
        st.write(f"**Aktuelles Signal für {symbol} im Zeitrahmen {timeframe}:** {last_signal}")

        if last_signal == 'Kaufen':
            st.success(f"Signal für {symbol}: **Kaufen** - Der 20-Tage-Durchschnitt liegt über dem 50-Tage-Durchschnitt.")
        elif last_signal == 'Verkaufen':
            st.warning(f"Signal für {symbol}: **Verkaufen** - Der 20-Tage-Durchschnitt liegt unter dem 50-Tage-Durchschnitt.")
        else:
            st.info(f"Signal für {symbol}: **Halten** - Keine klare Richtung.")

        entry_price = data['close'].iloc[-1]
        stop_loss = entry_price * 0.95
        position_size = calculate_position_size(account_size, risk_percentage, entry_price, stop_loss)
        st.success(f"Empfohlene Positionsgröße für {symbol}: {position_size:.4f} {symbol.split('/')[0]}")

# Anfänger-freundliche Funktionen
st.header("🎓 Anfänger-Bereich")

# Trendstärke Anzeige
for symbol in selected_symbols:
    data = fetch_market_data(symbol, timeframe, limit)
    
    # Sicherstellen, dass die 'close'-Spalte vorhanden ist
    if 'close' not in data.columns:
        st.error(f"Fehler: 'close'-Spalte fehlt in den Daten für {symbol}")
        continue
    
    trend_strength = calculate_trend_strength(data)
    st.subheader(f"📊 Trendstärke für {symbol}")
    trend_strength_value = trend_strength.iloc[-1]
    st.write(f"Aktuelle Trendstärke: {trend_strength_value:.1f}%")

    if trend_strength_value > 70:
        st.success("Starker Trend - gute Bedingungen für Trendfolge-Strategien")
    elif trend_strength_value < 30:
        st.warning("Schwacher Trend - besser auf klare Signale warten")
    else:
        st.info("Moderater Trend - normale Marktbedingungen")

# Support und Resistance
for symbol in selected_symbols:
    data = fetch_market_data(symbol, timeframe, limit)
    
    # Sicherstellen, dass die 'high' und 'low'-Spalten vorhanden sind
    if 'high' not in data.columns or 'low' not in data.columns:
        st.error(f"Fehler: 'high' oder 'low'-Spalte fehlt in den Daten für {symbol}")
        continue
    
    support, resistance = find_support_resistance(data)
    st.subheader(f"🎯 Support & Resistance Levels für {symbol}")
    st.write(f"Nächster Support-Level: {support:.2f}")
    st.write(f"Nächster Resistance-Level: {resistance:.2f}")

# Risikokalkulator
for symbol in selected_symbols:
    st.subheader(f"💰 Risikokalkulator für {symbol}")
    col1, col2 = st.columns(2)
    with col1:
        account_size = st.number_input("Kontostand (USDT)", min_value=100.0, value=1000.0, step=100.0, key=f'account_size_input_{symbol}')
        risk_percentage = st.number_input("Risiko pro Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f'risk_percentage_{symbol}')
    with col2:
        entry_price = st.number_input("Einstiegspreis", min_value=0.1, value=float(data['close'].iloc[-1]), step=0.1, key=f'entry_price_{symbol}')
        stop_loss = st.number_input("Stop-Loss Preis", min_value=0.1, value=float(data['close'].iloc[-1])*0.95, step=0.1, key=f'stop_loss_{symbol}')

    position_size = calculate_position_size(account_size, risk_percentage, entry_price, stop_loss)
    st.success(f"Empfohlene Positionsgröße: {position_size:.4f} {symbol.split('/')[0]}")

# Grundlagen-Erklärungen
st.subheader("📚 Grundlagen für Anfänger")
st.markdown("""
### 🔑 Wichtige Konzepte:

1. **Trendfolge**
   - Ein Trend ist dein Freund - handle in Richtung des übergeordneten Trends
   - Kaufe in Aufwärtstrends bei Rücksetzern
   - Verkaufe in Abwärtstrends bei Erholungen

2. **Support & Resistance**
   - Support: Preisbereich, wo Käufer aktiv werden
   - Resistance: Preisbereich, wo Verkäufer aktiv werden
   - Diese Levels sind wichtige Entscheidungspunkte

3. **Risikomanagement**
   - Nutze immer einen Stop-Loss
   - Riskiere nie mehr als 1-2% deines Kapitals pro Trade
   - Plane deine Trades im Voraus

4. **Grundlegende Indikatoren**
   - RSI: Zeigt überkaufte/überverkaufte Bereiche
   - Moving Averages: Zeigen Trends und mögliche Unterstützung
   - Volumen: Bestätigt Preisbewegungen

### ⚠️ Häufige Anfängerfehler vermeiden:
- Nicht ohne Stop-Loss handeln
- Nicht einem Verlust hinterherlaufen
- Nicht zu große Positionen eingehen
- Nicht gegen den Trend handeln
- Nicht zu oft handeln - Qualität über Quantität
""")

# Marktphasen-Erklärung
st.subheader("🌊 Aktuelle Marktphase")
for symbol in selected_symbols:
    data = fetch_market_data(symbol, timeframe, limit)
    
    # Sicherstellen, dass die 'close'-Spalte vorhanden ist
    if 'close' not in data.columns:
        st.error(f"Fehler: 'close'-Spalte fehlt in den Daten für {symbol}")
        continue
    
    # Berechne RSI
    data['RSI'] = calculate_rsi(data['close'])
    rsi = data['RSI'].iloc[-1]
    
    # Berechne Volatilität
    volatility = calculate_volatility(data).iloc[-1] * 100
    
    # Berechne Trendstärke
    trend_strength_value = calculate_trend_strength(data).iloc[-1]
    
    def determine_market_phase():
        if trend_strength_value > 60:
            if rsi > 60:
                return "Starker Aufwärtstrend", "Trendfolge-Strategien bevorzugen, auf Rücksetzer achten"
            else:
                return "Beginnender Aufwärtstrend", "Auf Ausbrüche nach oben achten"
        elif trend_strength_value < 40:
            if rsi < 40:
                return "Starker Abwärtstrend", "Vorsichtig sein, keine voreiligen Käufe"
            else:
                return "Beginnender Abwärtstrend", "Gewinne absichern, Stop-Loss eng setzen"
        else:
            return "Seitwärtsphase", "Auf Range-Trading setzen, Unterstützung und Widerstände beachten"

    phase, recommendation = determine_market_phase()
    st.write(f"**Aktuelle Phase für {symbol}:** {phase}")
    st.write(f"**Empfehlung:** {recommendation}")

# Weltmarkt-Analyse
st.header("🌍 Globale Marktübersicht")

# Weltmarkt-Daten anzeigen
st.subheader("📊 Wichtige Marktindikatoren")
market_data = fetch_global_market_data()

# Erstelle drei Spalten für die Marktdaten
col1, col2, col3 = st.columns(3)

# Funktion für farbige Darstellung der Änderungen
def format_change(change):
    if change is None:
        return "Keine Daten"
    color = "green" if change >= 0 else "red"
    return f"<span style='color: {color}'>{change:+.2f}%</span>"

# Verteile die Marktdaten auf die Spalten
indicators = list(market_data.items())
for i, (name, data) in enumerate(indicators):
    with [col1, col2, col3][i % 3]:
        st.markdown(f"**{name}**")
        if data['price'] is not None:
            st.markdown(f"Preis: {data['price']:.2f}")
            st.markdown(f"Änderung: {format_change(data['change'])}", unsafe_allow_html=True)
        else:
            st.markdown("Keine Daten verfügbar")

# Fear & Greed Index
st.subheader("😨 Fear & Greed Index")
fear_greed = fetch_fear_greed_index()
col1, col2 = st.columns(2)

with col1:
    st.metric("Fear & Greed Wert", fear_greed['value'])
with col2:
    st.write(f"Status: **{fear_greed['classification']}**")
    
    # Interpretationen basierend auf dem Index
    if int(fear_greed['value']) < 25:
        st.warning("Extreme Angst - Historisch oft eine gute Kaufgelegenheit")
    elif int(fear_greed['value']) > 75:
        st.warning("Extreme Gier - Vorsicht geboten, mögliche Korrektur")
    else:
        st.info("Neutraler Markt - Normale Handelsbedingungen")

# Marktkorrelationen und Einflüsse
st.subheader("🔄 Marktkorrelationen")
st.markdown("""
### Aktuelle Markteinflüsse:

1. **Aktienmarkt (S&P 500)**
   - Positive Korrelation mit Krypto in Risiko-On Phasen
   - Wichtiger Indikator für globales Risikosentiment
   
2. **US Dollar Index**
   - Negative Korrelation mit Krypto
   - Starker Dollar = Schwächere Kryptowährungen
   
3. **Gold**
   - Traditioneller "sicherer Hafen"
   - Korrelation mit Bitcoin in Krisenzeiten
   
4. **Volatilitätsindex (VIX)**
   - "Angstbarometer" des Marktes
   - Hoher VIX = Erhöhte Marktängste
   
5. **Anleihenrenditen**
   - Einfluss auf Kapitalflüsse
   - Höhere Renditen können Krypto-Investments weniger attraktiv machen
""")

# Handelsempfehlungen basierend auf Makrodaten
st.subheader("📈 Makroökonomische Handelsempfehlungen")

def generate_macro_recommendations(market_data, fear_greed_value):
    recommendations = []
    
    # Analyse des Marktumfelds
    try:
        vix = market_data['Volatilitätsindex']['price']
        if vix > 30:
            recommendations.append("⚠️ Hohe Marktvolatilität - Reduzierte Positionsgrößen empfohlen")
        elif vix < 15:
            recommendations.append("✅ Niedrige Volatilität - Normale Positionsgrößen möglich")
            
        # Dollar-Analyse
        if market_data['US Dollar Index']['change'] > 0.5:
            recommendations.append("📉 Starker Dollar - Vorsicht bei Krypto-Longpositionen")
        elif market_data['US Dollar Index']['change'] < -0.5:
            recommendations.append("📈 Schwacher Dollar - Positiv für Kryptowährungen")
            
        # Gesamtmarkt-Sentiment
        if int(fear_greed_value) < 20:
            recommendations.append("💡 Extremes Angst-Level - Antizyklische Kaufgelegenheiten beobachten")
        elif int(fear_greed_value) > 80:
            recommendations.append("⚠️ Extreme Gier - Gewinne absichern, vorsichtiger agieren")
            
    except:
        recommendations.append("⚠️ Einige Marktdaten nicht verfügbar - Vorsichtig handeln")
    
    return recommendations

macro_recommendations = generate_macro_recommendations(market_data, fear_greed['value'])
for rec in macro_recommendations:
    st.markdown(rec)

# Nachrichtenanalyse und Medieneinflüsse
st.header("📰 Marktrelevanzte Nachrichten & Medienanalyse")

# Nachrichten abrufen
news_data = fetch_crypto_news()

# Krypto-Nachrichten
st.subheader("🔗 Aktuelle Krypto-Nachrichten")
if news_data['crypto']:
    for article in news_data['crypto']:
        with st.expander(article.get('title', 'Keine Überschrift')):
            description = article.get('description', 'Keine Beschreibung verfügbar')
            sentiment, score = analyze_news_sentiment(description)
            impact = analyze_market_impact(article)
            
            st.write(description)
            st.write(f"Quelle: **{article.get('source', {}).get('name', 'Unbekannt')}** | Datum: {article.get('publishedAt', '')[:10]}")
            
            # Sentiment und Markteinfluss anzeigen
            col1, col2 = st.columns(2)
            with col1:
                sentiment_color = "green" if score > 0 else "red" if score < 0 else "gray"
                st.markdown(f"Sentiment: <span style='color: {sentiment_color}'>{sentiment} ({score:.1f}%)</span>", 
                          unsafe_allow_html=True)
            with col2:
                impact_text = "⬆️ Positiv" if impact > 0 else "⬇️ Negativ" if impact < 0 else "➡️ Neutral"
                st.write(f"Möglicher Markteinfluss: {impact_text}")
            
            if article.get('url'):
                st.markdown(f"[Vollständiger Artikel]({article['url']})")
else:
    st.info("Keine aktuellen Krypto-Nachrichten verfügbar")

# Wirtschaftsnachrichten
st.subheader("📊 Wichtige Wirtschaftsnachrichten")
if news_data['economic']:
    for article in news_data['economic']:
        with st.expander(article.get('title', 'Keine Überschrift')):
            description = article.get('description', 'Keine Beschreibung verfügbar')
            sentiment, score = analyze_news_sentiment(description)
            st.write(description)
            st.write(f"Quelle: **{article.get('source', {}).get('name', 'Unbekannt')}** | Datum: {article.get('publishedAt', '')[:10]}")
            
            sentiment_color = "green" if score > 0 else "red" if score < 0 else "gray"
            st.markdown(f"Sentiment: <span style='color: {sentiment_color}'>{sentiment} ({score:.1f}%)</span>",
                      unsafe_allow_html=True)
            
            if article.get('url'):
                st.markdown(f"[Vollständiger Artikel]({article['url']})")
else:
    st.info("Keine aktuellen Wirtschaftsnachrichten verfügbar")

# Gesamtanalyse des Nachrichtensentiments
st.subheader("📈 Mediensentiment-Analyse")
st.markdown("""
### 🎯 Interpretation der Nachrichten:

1. **Krypto-spezifische Faktoren**
   - Regulatorische Entwicklungen
   - Technologische Fortschritte
   - Institutionelle Adoption
   - Sicherheitsvorfälle

2. **Makroökonomische Einflüsse**
   - Zinspolitik der Zentralbanken
   - Inflationsentwicklung
   - Geopolitische Ereignisse
   - Wirtschaftswachstum

3. **Marktsentiment**
   - Medienwahrnehmung
   - Soziale Medien Trends
   - Institutionelles Interesse
   - Retail-Interesse
""")

# Hinweise zur Nachrichteninterpretation
st.info("""
**💡 Tipps zur Nachrichtenanalyse:**
- Fokussiere auf verifizierte Nachrichtenquellen
- Beachte den zeitlichen Kontext der Nachrichten
- Unterscheide zwischen kurzfristigen und langfristigen Auswirkungen
- Berücksichtige mehrere Perspektiven
- Sei vorsichtig bei extremen Schlagzeilen
""")

# Footer
st.markdown("---")
st.markdown(" 2025 Krypto Marktanalyse App")