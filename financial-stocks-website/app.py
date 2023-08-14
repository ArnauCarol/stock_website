import yfinance as yf
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stocks/<symbol>')
def stocks(symbol):
    if symbol == 'sp500':
        stock_symbol = '^GSPC'
    elif symbol == 'ibex35':
        stock_symbol = '^IBEX'
    elif symbol == 'nasdq':
        stock_symbol = '^IXIC'
    else:
        return "Invalid stock symbol."

    stock_data = yf.download(stock_symbol, period='10y')
    stock_dates = stock_data.index.strftime('%Y-%m-%d').tolist()
    stock_prices = stock_data['Close'].tolist()

    return render_template('stocks.html', stock_symbol=stock_symbol, stock_dates=stock_dates, stock_prices=stock_prices)

if __name__ == '__main__':
    app.run(debug=True)
