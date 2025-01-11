import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import time
import threading
import concurrent.futures
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("BITPANDA_API_KEY")
api_secret = os.getenv("BITPANDA_API_SECRET")

if not api_key or not api_secret:
    raise ValueError("API key and secret must be set in .env file")

# Initialize exchange
exchange = ccxt.bitpanda({
    'apiKey': api_key,
    'enableRateLimit': True,
    'timeout': 30000,
    'urls': {
        'api': 'https://api.bitpanda.com/v1'
    },
    'headers': {
        'X-Api-Key': api_key,
        'Accept': 'application/json'
    }
})

# Test connection by fetching wallets
try:
    st.write("Testing connection to Bitpanda...")
    headers = {
        'X-Api-Key': api_key,
        'Accept': 'application/json'
    }
    
    response = requests.get(
        'https://api.bitpanda.com/v1/wallets',
        headers=headers
    )
    
    if response.status_code == 200:
        st.write("Successfully connected to Bitpanda")
        data = response.json()
        if 'data' in data:
            wallets = data['data']
            st.write("\nAvailable wallets:")
            for wallet in wallets:
                if isinstance(wallet, dict) and 'attributes' in wallet:
                    attrs = wallet['attributes']
                    st.write(f"- {attrs.get('cryptocoin_symbol', 'Unknown')}: Balance = {attrs.get('balance', '0.0')}")
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
except Exception as e:
    st.error(f"Error connecting to Bitpanda: {str(e)}")
    raise

# Define trading strategy
class TradingBot:
    def __init__(self, symbol, timeframe, risk_percentage, asset_type=None, params=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percentage = risk_percentage
        self.base_currency = symbol.split('/')[0] if '/' in symbol else symbol
        self.headers = {
            'X-Api-Key': api_key,
            'Accept': 'application/json'
        }
        
        # Trading Parameter für Scalping/CFD
        self.short_ma = 5    # Sehr kurzer MA für schnelle Signale
        self.long_ma = 15    # Kurzer MA für Trendbestätigung
        self.rsi_period = 7  # Kürzerer RSI für schnellere Signale
        self.rsi_overbought = 75  # Angepasst für volatile Märkte
        self.rsi_oversold = 25
        
        # CFD-spezifische Parameter
        self.leverage = 5  # 5x Hebel
        self.stop_loss_pct = 1.0  # 1% Stop Loss
        self.take_profit_pct = 2.0  # 2% Take Profit
        
        # Asset-spezifische Parameter
        if asset_type and params:
            self.leverage = params['leverage']
            self.stop_loss_pct = params['stop_loss']
            self.take_profit_pct = params['take_profit']
        
        # Performance-Optimierung
        self.cache = {}  # Cache für Berechnungen
        self.last_price = None
        self.position = None
        
        # Verify the symbol exists
        self.verify_trading_pair()

    def verify_trading_pair(self):
        """Verify that we can trade this symbol"""
        try:
            response = requests.get(
                'https://api.bitpanda.com/v1/wallets',
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    available_symbols = [
                        wallet['attributes']['cryptocoin_symbol']
                        for wallet in data['data']
                        if isinstance(wallet, dict) and 'attributes' in wallet
                    ]
                    
                    if self.base_currency not in available_symbols:
                        raise ValueError(
                            f"Symbol {self.base_currency} not found. Available symbols: {', '.join(available_symbols)}"
                        )
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Error verifying trading pair: {str(e)}")
            raise

    def fetch_ohlcv(self):
        try:
            # Get historical data
            response = requests.get(
                f'https://api.bitpanda.com/v3/candlesticks/{self.symbol}',
                headers=self.headers,
                params={
                    'unit': self.timeframe,
                    'period': '1', # Get last period
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch OHLCV data: {response.text}")
                
            data = response.json()
            
            # Ensure we have data
            if not data or 'data' not in data or not data['data']:
                st.write(f"No OHLCV data available for {self.symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Sort by timestamp
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
            
            # Ensure we have enough data
            if len(df) < self.long_ma:
                st.write(f"Not enough data points for {self.symbol} (need {self.long_ma}, got {len(df)})")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            st.error(f"Error fetching OHLCV data: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        try:
            # Check if DataFrame is empty
            if df.empty:
                st.write(f"No data available for {self.symbol}")
                return df
                
            # Ensure we have enough data
            if len(df) < self.long_ma:
                st.write(f"Not enough data points for {self.symbol}")
                return df
                
            # Convert to numpy array for faster calculations
            closes = df['close'].values
            
            # Moving Averages
            df['ma_short'] = pd.Series(
                np.convolve(closes, np.ones(self.short_ma)/self.short_ma, mode='valid'),
                index=df.index[self.short_ma-1:]
            ).fillna(method='ffill')
            
            df['ma_long'] = pd.Series(
                np.convolve(closes, np.ones(self.long_ma)/self.long_ma, mode='valid'),
                index=df.index[self.long_ma-1:]
            ).fillna(method='ffill')
            
            # RSI
            delta = np.diff(closes)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gains).rolling(window=self.rsi_period).mean().fillna(0)
            avg_loss = pd.Series(losses).rolling(window=self.rsi_period).mean().fillna(0)
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)  # Neutrale RSI-Werte für NaN
            
            # Bollinger Bands
            df['sma'] = df['close'].rolling(window=20).mean().fillna(method='ffill')
            df['std'] = df['close'].rolling(window=20).std().fillna(method='ffill')
            df['bb_upper'] = df['sma'] + (df['std'] * 2)
            df['bb_lower'] = df['sma'] - (df['std'] * 2)
            
            # MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Momentum (3 Perioden)
            df['momentum'] = df['close'].pct_change(periods=3).fillna(0)
            
            # Volatilität (10 Perioden)
            df['volatility'] = df['close'].pct_change().rolling(window=10).std().fillna(0) * np.sqrt(10)
            
            # Debug-Ausgabe
            st.write("\nIndicator values for last row:")
            st.write(f"MA Short: {df['ma_short'].iloc[-1]:.2f}")
            st.write(f"MA Long: {df['ma_long'].iloc[-1]:.2f}")
            st.write(f"RSI: {df['rsi'].iloc[-1]:.2f}")
            st.write(f"MACD: {df['macd'].iloc[-1]:.2f}")
            st.write(f"MACD Signal: {df['macd_signal'].iloc[-1]:.2f}")
            st.write(f"Momentum: {df['momentum'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            raise

    def generate_signal(self, df):
        try:
            # Initialize signal column
            df['signal'] = 0
            
            for i in range(1, len(df)):
                # Vereinfachte Bedingungen für Tests
                
                # Kaufsignal bei steigendem Trend
                if df['momentum'].iloc[i] > 0:
                    df.loc[df.index[i], 'signal'] = 1
                
                # Verkaufssignal bei fallendem Trend
                elif df['momentum'].iloc[i] < 0:
                    df.loc[df.index[i], 'signal'] = -1
            
            # Debug-Ausgabe für Signale
            st.write("\nGenerated signals:")
            st.write(df[['momentum', 'signal']].tail())
            
            return df
            
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
            raise

    def analyze_market(self, df):
        try:
            if df.empty:
                return
                
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Preisänderung berechnen
            price_change_5min = ((latest['close'] - prev['close']) / prev['close']) * 100
            
            st.write(f"\n📊 Marktanalyse für {self.symbol}:")
            st.write(f"💰 Aktueller Preis: €{latest['close']:.2f}")
            st.write(f"📈 5min Preisänderung: {price_change_5min:.2f}%")
            
            # Technische Indikatoren
            st.write("\n📈 Technische Indikatoren:")
            st.write(f"RSI: {latest['rsi']:.2f}")
            st.write(f"Momentum: {latest['momentum']*100:.2f}%")
            
            # Marktbedingungen analysieren
            st.write("\n🔍 Marktbedingungen:")
            
            # RSI Bedingungen
            if latest['rsi'] > self.rsi_overbought:
                st.write("⚠️ Überkauft - Verkaufssignal möglich")
            elif latest['rsi'] < self.rsi_oversold:
                st.write("💡 Überverkauft - Kaufsignal möglich")
            
            # Bollinger Bands
            if latest['close'] > latest['bb_upper']:
                st.write("⚠️ Preis über oberem Bollinger Band - Überkauft")
            elif latest['close'] < latest['bb_lower']:
                st.write("💡 Preis unter unterem Bollinger Band - Überverkauft")
            
            # MACD
            if latest['macd'] > latest['macd_signal']:
                st.write("📈 MACD über Signallinie - Bullischer Trend")
            else:
                st.write("📉 MACD unter Signallinie - Bärischer Trend")
            
        except Exception as e:
            st.error(f"Error analyzing market: {str(e)}")

    def calculate_position_size(self, current_price):
        """Berechnet die optimale Positionsgröße für CFD-Trading"""
        try:
            # Get account balance
            response = requests.get(
                'https://api.bitpanda.com/v1/fiatwallets',
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get balance: {response.text}")
                
            data = response.json()
            eur_wallet = next(
                (w for w in data['data']
                 if w['attributes']['fiat_symbol'] == 'EUR'),
                None
            )
            
            if not eur_wallet:
                raise Exception("No EUR wallet found")
                
            account_balance = float(eur_wallet['attributes']['balance'])
            
            # Berechne Stop-Loss in Punkten
            stop_loss_points = current_price * (self.stop_loss_pct / 100)
            
            # Maximaler Verlust pro Trade
            max_loss = account_balance * (self.risk_percentage / 100)
            
            # Positionsgröße unter Berücksichtigung des Hebels
            position_size = (max_loss / stop_loss_points) * self.leverage
            
            return position_size
            
        except Exception as e:
            st.error(f"Error calculating position size: {str(e)}")
            raise

    def execute_trade(self, signal):
        try:
            if signal == 0:
                st.write("Kein Handelssignal vorhanden")
                return
                
            action = "buy" if signal == 1 else "sell"
            st.write(f"\n🔄 Executing {action} signal for {self.symbol}")
            
            # Get current price
            try:
                price_response = requests.get(
                    f'https://api.bitpanda.com/v1/ticker',
                    headers=self.headers
                )
                st.write(f"Price API Response Status: {price_response.status_code}")
                st.write(f"Price API Response: {price_response.text[:200]}")  # First 200 chars
                
                if price_response.status_code != 200:
                    st.error(f"Failed to get price: {price_response.text}")
                    return
                    
                price_data = price_response.json()
                current_price = float(price_data.get(self.base_currency, {}).get('EUR', 0))
                
                if current_price == 0:
                    st.error(f"Could not get price for {self.symbol}")
                    return
                    
                st.write(f"Current Price: {current_price}")
                
                # Berechne Position Size
                position_size = self.calculate_position_size(current_price)
                st.write(f"Calculated Position Size: {position_size}")
                
                # Berechne Stop-Loss und Take-Profit
                if signal == 1:  # Long
                    stop_loss = current_price * (1 - self.stop_loss_pct/100)
                    take_profit = current_price * (1 + self.take_profit_pct/100)
                else:  # Short
                    stop_loss = current_price * (1 + self.stop_loss_pct/100)
                    take_profit = current_price * (1 - self.take_profit_pct/100)
                
                # Execute order
                try:
                    order_data = {
                        'type': 'market',
                        'side': action,
                        'leverage': self.leverage,
                        'size': position_size,
                        'symbol': self.symbol,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                    st.write("\n📊 Trade Details:")
                    st.write(f"Position Size: {position_size:.8f}")
                    st.write(f"Leverage: {self.leverage}x")
                    st.write(f"Entry Price: €{current_price:.2f}")
                    st.write(f"Stop Loss: €{stop_loss:.2f}")
                    st.write(f"Take Profit: €{take_profit:.2f}")
                    
                    # Sende Order an Bitpanda
                    order_response = requests.post(
                        'https://api.bitpanda.com/v1/trading/orders',
                        headers=self.headers,
                        json=order_data
                    )
                    
                    st.write(f"\nOrder API Response Status: {order_response.status_code}")
                    st.write(f"Order API Response: {order_response.text[:200]}")  # First 200 chars
                    
                    if order_response.status_code == 200:
                        st.success(f"✅ Trade executed successfully!")
                        # Speichere Position für Tracking
                        self.position = {
                            'type': action,
                            'entry_price': current_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'leverage': self.leverage
                        }
                    else:
                        st.error(f"❌ Failed to execute trade: {order_response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error executing order: {str(e)}")
                    
            except Exception as e:
                st.error(f"❌ Error getting price: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error in execute_trade: {str(e)}")

    def run(self):
        try:
            st.write(f"\nStarting trading bot for {self.symbol}...")
            
            # Fetch and process data
            df = self.fetch_ohlcv()
            
            if df.empty:
                st.write(f"No data available for {self.symbol}, skipping...")
                return
            
            st.write(f"Fetched {len(df)} data points")
            
            st.write("\nCalculating technical indicators...")
            df = self.calculate_indicators(df)
            
            if df.empty:
                st.write(f"Could not calculate indicators for {self.symbol}, skipping...")
                return
            
            # Generate trading signals
            df = self.generate_signal(df)
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1] if not df.empty else 0
            st.write(f"Latest signal for {self.symbol}: {latest_signal}")
            
            # Analyze market conditions
            self.analyze_market(df)
            
            # Execute trade if signal exists
            if latest_signal != 0:
                self.execute_trade(latest_signal)
            else:
                st.write("No trading signal generated")
        
        except Exception as e:
            st.error(f"Error running trading bot: {str(e)}")
            raise

class MultiAssetTradingBot:
    def __init__(self, timeframe='1m', risk_percentage=1):
        self.timeframe = timeframe
        self.risk_percentage = risk_percentage
        self.headers = {
            'X-API-KEY': api_key,
            'Accept': 'application/json'
        }
        self.running = False
        self.bots = {}
        self.lock = threading.Lock()
        self.last_trade_time = {}
        
        # Performance-Optimierung
        self.update_interval = 30  # 30 Sekunden Update-Intervall
        self.min_trade_interval = 30  # Mindestens 30 Sekunden zwischen Trades
        self.max_concurrent_requests = 10  # Mehr parallele Anfragen
        
        # Asset-spezifische Parameter
        self.asset_params = {
            'crypto': {
                'leverage': 5,
                'min_volatility': 0.5,
                'stop_loss': 1.0,
                'take_profit': 2.0
            },
            'stocks': {
                'leverage': 3,
                'min_volatility': 0.3,
                'stop_loss': 0.8,
                'take_profit': 1.6
            },
            'etfs': {
                'leverage': 2,
                'min_volatility': 0.2,
                'stop_loss': 0.5,
                'take_profit': 1.0
            },
            'metals': {
                'leverage': 4,
                'min_volatility': 0.4,
                'stop_loss': 0.7,
                'take_profit': 1.4
            }
        }

    def get_all_assets(self):
        """Holt alle verfügbaren Assets von Bitpanda"""
        try:
            assets = {
                'crypto': [],
                'stocks': [],
                'etfs': [],
                'metals': []
            }
            
            # Kryptowährungen
            crypto_response = requests.get(
                'https://api.bitpanda.com/v1/cryptocoins',
                headers=self.headers
            )
            if crypto_response.status_code == 200:
                cryptos = crypto_response.json()
                assets['crypto'] = [c['attributes']['symbol'] for c in cryptos['data']]
            
            # Aktien
            stocks_response = requests.get(
                'https://api.bitpanda.com/v1/stocks',
                headers=self.headers
            )
            if stocks_response.status_code == 200:
                stocks = stocks_response.json()
                assets['stocks'] = [s['attributes']['symbol'] for s in stocks['data']]
            
            # ETFs
            etfs_response = requests.get(
                'https://api.bitpanda.com/v1/etfs',
                headers=self.headers
            )
            if etfs_response.status_code == 200:
                etfs = etfs_response.json()
                assets['etfs'] = [e['attributes']['symbol'] for e in etfs['data']]
            
            # Metalle
            metals_response = requests.get(
                'https://api.bitpanda.com/v1/metals',
                headers=self.headers
            )
            if metals_response.status_code == 200:
                metals = metals_response.json()
                assets['metals'] = [m['attributes']['symbol'] for m in metals['data']]
            
            return assets
            
        except Exception as e:
            st.error(f"Error getting assets: {str(e)}")
            return {}

    def start_trading(self, status_callback=None):
        self.running = True
        if status_callback:
            status_callback("\n🚀 Starting High-Frequency Multi-Asset Trading Bot")
            status_callback(f"⚡ Ultra-Fast Mode: {self.timeframe} timeframe")
            status_callback(f"💰 Risk per Trade: {self.risk_percentage}%")
            status_callback("⚠️ Trading with Leverage - High Risk, High Reward")
        else:
            st.write("\n🚀 Starting High-Frequency Multi-Asset Trading Bot")
            st.write(f"⚡ Ultra-Fast Mode: {self.timeframe} timeframe")
            st.write(f"💰 Risk per Trade: {self.risk_percentage}%")
            st.write("⚠️ Trading with Leverage - High Risk, High Reward")
        
        # Asset-Tracking für Performance-Analyse
        self.performance_tracker = {
            'crypto': {'trades': 0, 'profit': 0},
            'stocks': {'trades': 0, 'profit': 0},
            'etfs': {'trades': 0, 'profit': 0},
            'metals': {'trades': 0, 'profit': 0}
        }
        
        while self.running:
            try:
                # Hole alle verfügbaren Assets
                all_assets = self.get_all_assets()
                total_assets = sum(len(assets) for assets in all_assets.values())
                if status_callback:
                    status_callback(f"\n📊 Analyzing {total_assets} assets across all markets")
                else:
                    st.write(f"\n📊 Analyzing {total_assets} assets across all markets")
                
                # Verarbeite alle Asset-Typen parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                    futures = []
                    
                    for asset_type, assets in all_assets.items():
                        if status_callback:
                            status_callback(f"\n💹 {asset_type.upper()} Markets:")
                        else:
                            st.write(f"\n💹 {asset_type.upper()} Markets:")
                        
                        for symbol in assets:
                            bot_key = f"{asset_type}_{symbol}"
                            
                            if bot_key not in self.bots:
                                if status_callback:
                                    status_callback(f"➕ Adding {symbol} to watchlist")
                                else:
                                    st.write(f"➕ Adding {symbol} to watchlist")
                                self.bots[bot_key] = TradingBot(
                                    symbol,
                                    self.timeframe,
                                    self.risk_percentage,
                                    asset_type=asset_type,
                                    params=self.asset_params[asset_type]
                                )
                                self.last_trade_time[bot_key] = datetime.now() - timedelta(seconds=30)
                            
                            # Prüfe Mindestabstand zwischen Trades
                            time_since_last_trade = datetime.now() - self.last_trade_time[bot_key]
                            if time_since_last_trade.total_seconds() >= self.min_trade_interval:
                                futures.append(
                                    executor.submit(
                                        self.process_asset,
                                        bot_key,
                                        symbol,
                                        asset_type,
                                        status_callback
                                    )
                                )
                    
                    # Warte auf Ergebnisse und sammle Performance-Daten
                    completed_futures = concurrent.futures.wait(
                        futures,
                        timeout=self.update_interval
                    )
                    
                    # Analysiere Ergebnisse
                    self.analyze_performance(status_callback)
                
                # Kurze Pause für API-Limits
                time.sleep(1)
                
            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error in main trading loop: {str(e)}")
                else:
                    st.error(f"❌ Error in main trading loop: {str(e)}")
                time.sleep(5)

    def process_asset(self, bot_key, symbol, asset_type, status_callback=None):
        """Verarbeitet ein einzelnes Asset und trackt die Performance"""
        try:
            if status_callback:
                status_callback(f"\n🔄 Analyzing {symbol}")
            else:
                st.write(f"\n🔄 Analyzing {symbol}")
            
            # Führe Trading-Logik aus
            result = self.bots[bot_key].run()
            
            # Update Performance-Tracking
            if result and 'profit' in result:
                with self.lock:
                    self.performance_tracker[asset_type]['trades'] += 1
                    self.performance_tracker[asset_type]['profit'] += result['profit']
            
            self.last_trade_time[bot_key] = datetime.now()
            
        except Exception as e:
            if status_callback:
                status_callback(f"❌ Error processing {symbol}: {str(e)}")
            else:
                st.error(f"❌ Error processing {symbol}: {str(e)}")

    def analyze_performance(self, status_callback=None):
        """Analysiert die Performance aller Assets"""
        try:
            if status_callback:
                status_callback("\n📈 Performance Analysis:")
            else:
                st.write("\n📈 Performance Analysis:")
            
            total_trades = 0
            total_profit = 0
            
            for asset_type, stats in self.performance_tracker.items():
                trades = stats['trades']
                profit = stats['profit']
                total_trades += trades
                total_profit += profit
                
                if status_callback:
                    status_callback(f"\n{asset_type.upper()}:")
                    status_callback(f"Trades: {trades}")
                    status_callback(f"Profit: €{profit:.2f}")
                    if trades > 0:
                        status_callback(f"Average Profit per Trade: €{profit/trades:.2f}")
                else:
                    st.write(f"\n{asset_type.upper()}:")
                    st.write(f"Trades: {trades}")
                    st.write(f"Profit: €{profit:.2f}")
                    if trades > 0:
                        st.write(f"Average Profit per Trade: €{profit/trades:.2f}")
            
            if status_callback:
                status_callback("\n📊 Overall Performance:")
                status_callback(f"Total Trades: {total_trades}")
                status_callback(f"Total Profit: €{total_profit:.2f}")
                if total_trades > 0:
                    status_callback(f"Overall Average Profit per Trade: €{total_profit/total_trades:.2f}")
            else:
                st.write("\n📊 Overall Performance:")
                st.write(f"Total Trades: {total_trades}")
                st.write(f"Total Profit: €{total_profit:.2f}")
                if total_trades > 0:
                    st.write(f"Overall Average Profit per Trade: €{total_profit/total_trades:.2f}")
            
            # Identifiziere beste Performance
            best_asset = max(
                self.performance_tracker.items(),
                key=lambda x: x[1]['profit']
            )
            if status_callback:
                status_callback(f"\n🏆 Best Performing Asset Type: {best_asset[0].upper()}")
                status_callback(f"Profit: €{best_asset[1]['profit']:.2f}")
            else:
                st.write(f"\n🏆 Best Performing Asset Type: {best_asset[0].upper()}")
                st.write(f"Profit: €{best_asset[1]['profit']:.2f}")
            
        except Exception as e:
            if status_callback:
                status_callback(f"Error analyzing performance: {str(e)}")
            else:
                st.error(f"Error analyzing performance: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Create multi-asset trading bot with 1-minute timeframe
        multi_bot = MultiAssetTradingBot(timeframe='1m', risk_percentage=1)
        
        st.write("\n⚠️ TEST MODE: Trading with CFDs - High Risk, High Reward")
        st.write("Make sure you understand the risks before proceeding.")
        st.write("Press Ctrl+C to stop the bot at any time.")
        
        # Start trading in a separate thread
        trading_thread = threading.Thread(target=multi_bot.start_trading)
        trading_thread.start()
        
        st.write("\n⌨️  Press Ctrl+C to stop trading")
        
        try:
            while trading_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            st.write("\n🛑 Stopping trading bot...")
            multi_bot.running = False
            trading_thread.join()
            st.write("✅ Trading bot stopped successfully")
            
    except Exception as e:
        st.error(f"\n❌ Trading bot stopped due to error: {str(e)}")
