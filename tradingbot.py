import ccxt
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

# Flask App Setup
app = Flask(__name__)

# Binance API Setup
BINANCE_API_KEY = 'your_binance_api_key'
BINANCE_SECRET_KEY = 'your_binance_api_key_secret'
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

# CoinGecko News API
COINGECKO_NEWS_API_URL = "https://api.coingecko.com/api/v3/search/trending"

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Global Variables
ACTIVE_TRADES = []
ML_MODEL = None

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_bot():
    trading_pairs = ["ETHUSDT", "SOLUSDT", "AAVEUSDT"]
    interval = Client.KLINE_INTERVAL_1HOUR
    lookback = 365 * 2

    # Run the bot asynchronously
    def run_bot():
        while True:
            for pair in trading_pairs:
                trading_logic(pair, interval, lookback)
            time.sleep(60)

    import threading
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    return jsonify({"status": "Bot started with pairs: " + ", ".join(trading_pairs)})

@app.route('/train', methods=['POST'])
def train_model():
    pair = request.json.get('pair', 'ETHUSDT')
    lookback = request.json.get('lookback', 365 * 2)
    interval = Client.KLINE_INTERVAL_1HOUR
    df = fetch_historical_data(pair, interval, lookback)
    df = calculate_ta_indicators(df)

    # Prepare data for training
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['RSI', 'MACD', 'MACD_signal', 'OBV', 'ADX', 'Aroon_up', 'Aroon_down']
    X = df[features].dropna()
    y = df['target'].dropna()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    global ML_MODEL
    ML_MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
    ML_MODEL.fit(X_train, y_train)

    y_pred = ML_MODEL.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return jsonify({"status": "Model trained", "accuracy": accuracy})

@app.route('/stop', methods=['POST'])
def stop_bot():
    # Stopping logic (if implemented in future versions)
    return jsonify({"status": "Bot stopping functionality is not implemented yet."})

@app.route('/active-trades', methods=['GET'])
def active_trades():
    active_trades_data = []
    for trade in ACTIVE_TRADES:
        pair = trade['pair']
        entry_price = trade['entry_price']
        current_price = get_current_price(pair)
        leverage = trade['leverage']
        risk_percentage = trade['risk_percentage']
        side = trade['side']

        if side == "LONG":
            pnl_percentage = ((current_price - entry_price) / entry_price) * leverage * 100
        else:  # SHORT
            pnl_percentage = ((entry_price - current_price) / entry_price) * leverage * 100

        active_trades_data.append({
            "pair": pair,
            "entry_price": entry_price,
            "current_price": current_price,
            "leverage": leverage,
            "risk_percentage": risk_percentage,
            "side": side,
            "pnl_percentage": pnl_percentage,
            "trailing_stop_loss": trade.get('trailing_stop_loss', None),
            "trailing_take_profit": trade.get('trailing_take_profit', None)
        })

    return jsonify(active_trades_data)

# Helper Function to Fetch Current Price
def get_current_price(pair):
    ticker = binance_client.get_symbol_ticker(symbol=pair)
    return float(ticker['price'])

# Fetch Historical Data
def fetch_historical_data(pair, interval, lookback):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=lookback)).timestamp() * 1000)

    klines = binance_client.get_historical_klines(pair, interval, start_time, end_time)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# TA Indicators
def calculate_ta_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['Aroon_up'] = ta.aroon(df['high'], df['low'], length=14)['AROOND_14']
    df['Aroon_down'] = ta.aroon(df['high'], df['low'], length=14)['AROONU_14']
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    return df

# NLP-Based Sentiment Analysis
def analyze_sentiment(text):
    results = sentiment_pipeline(text)
    sentiment_score = 0
    for result in results:
        if result['label'] == 'POSITIVE':
            sentiment_score += result['score']
        elif result['label'] == 'NEGATIVE':
            sentiment_score -= result['score']
    return sentiment_score

# News Sentiment Analysis
def fetch_news_sentiment():
    response = requests.get(COINGECKO_NEWS_API_URL)
    if response.status_code == 200:
        articles = response.json().get('coins', [])
        titles = [coin['item']['name'] for coin in articles]
        combined_text = " ".join(titles)
        sentiment_score = analyze_sentiment(combined_text)
        return sentiment_score
    else:
        print("Error fetching news")
        return 0

# Risk Management
def calculate_risk(balance, sentiment, ta_signal):
    risk_factor = (sentiment + ta_signal) / 2
    if risk_factor > 0.5:
        risk_percentage = 0.05  # High confidence, use 5% of balance
    elif 0.2 < risk_factor <= 0.5:
        risk_percentage = 0.02  # Medium confidence, use 2% of balance
    else:
        risk_percentage = 0.01  # Low confidence, use 1% of balance

    return balance * risk_percentage

# Leverage Calculation
def determine_leverage(sentiment, ta_signal):
    confidence = (sentiment + ta_signal) / 2
    leverage = int(7 + (confidence * 13))  # Scales leverage between 7x and 20x
    return max(7, min(20, leverage))

# Trade Execution Function
def execute_trade(pair, side, quantity, leverage, stop_loss=None, take_profit=None):
    try:
        # Set leverage
        binance_client.futures_change_leverage(symbol=pair, leverage=leverage)

        if side == "LONG":
            order = binance_client.create_margin_order(
                symbol=pair, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity
            )
        elif side == "SHORT":
            order = binance_client.create_margin_order(
                symbol=pair, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity
            )

        entry_price = float(order['fills'][0]['price'])
        ACTIVE_TRADES.append({
            "pair": pair,
            "entry_price": entry_price,
            "leverage": leverage,
            "risk_percentage": 0.05 if side == "LONG" else 0.02,
            "side": side,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        })
        print(f"Trade executed: {order}")
    except Exception as e:
        print(f"Error executing trade: {e}")

# Main Trading Logic
def trading_logic(pair, interval, lookback):
    df = fetch_historical_data(pair, interval, lookback)
    df = calculate_ta_indicators(df)

    latest_data = df.iloc[-1]
    ta_signal = (latest_data['RSI'] > 50) + (latest_data['MACD'] > latest_data['MACD_signal']) + (latest_data['ADX'] > 25)
    ta_signal = ta_signal / 3  # Normalize to a 0-1 scale

    news_sentiment = fetch_news_sentiment()

    overall_sentiment = news_sentiment

    print(f"Pair: {pair} | TA Signal: {ta_signal} | News Sentiment: {news_sentiment}")

    balance = float(binance_client.get_asset_balance(asset='USDT')['free'])
    risk_amount = calculate_risk(balance, overall_sentiment, ta_signal)
    leverage = determine_leverage(overall_sentiment, ta_signal)
    quantity = risk_amount * leverage / float(latest_data['close'])

    stop_loss = float(latest_data['close']) * 0.98 if overall_sentiment > 0.5 else float(latest_data['close']) * 1.02
    take_profit = float(latest_data['close']) * 1.05 if overall_sentiment > 0.5 else float(latest_data['close']) * 0.95

    if overall_sentiment > 0.5 and ta_signal > 0.5:
        print("Bullish sentiment and TA detected. Executing LONG order.")
        execute_trade(pair, "LONG", quantity, leverage, stop_loss, take_profit)
    elif overall_sentiment < -0.5 and ta_signal < 0.5:
        print("Bearish sentiment and TA detected. Executing SHORT order.")
        execute_trade(pair, "SHORT", quantity, leverage, stop_loss, take_profit)
    else:
        print("Neutral sentiment. No action taken.")

if __name__ == "__main__":
    app.run(debug=True)

