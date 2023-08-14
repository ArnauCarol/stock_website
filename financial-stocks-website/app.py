import yfinance as yf
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stocks')
def stocks():
    stock_symbol = 'AAPL'  # Replace with your desired stock symbol
    stock_data = yf.download(stock_symbol, period='10y')
    stock_dates = stock_data.index.strftime('%Y-%m-%d').tolist()  # Convert Index to list of strings
    stock_prices = stock_data['Close'].tolist()

    return render_template('stocks.html', stock_symbol=stock_symbol, stock_dates=stock_dates, stock_prices=stock_prices)

if __name__ == '__main__':
    app.run(debug=True)
