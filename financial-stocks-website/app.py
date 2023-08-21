import sqlite3
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ibex35/<symbol>')
def ibex35(symbol):
    if symbol == 'ibex35':
        table_name = 'IBEX35'
        stock_symbol = '^IBEX'
    else:
        return "Invalid stock symbol."

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT date, close, volume FROM {table_name}')
    data = cursor.fetchall()
    conn.close()

    stock_dates = [row[0] for row in data]
    stock_prices = [row[1] for row in data]
    stock_volumes = [row[2] for row in data]

    return render_template('ibex35.html', stock_symbol=stock_symbol,
                           stock_dates=stock_dates, stock_prices=stock_prices,
                           stock_volumes=stock_volumes, table_name=table_name)
    
    
@app.route('/sp500/<symbol>')
def sp500(symbol):
    if symbol == 'sp500':
        table_name = 'SP500'
        stock_symbol = '^GSPC'
    else:
        return "Invalid stock symbol."

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT date, close, volume FROM {table_name}')
    data = cursor.fetchall()
    conn.close()

    stock_dates = [row[0] for row in data]
    stock_prices = [row[1] for row in data]
    stock_volumes = [row[2] for row in data]

    return render_template('sp500.html', stock_symbol=stock_symbol,
                           stock_dates=stock_dates, stock_prices=stock_prices,
                           stock_volumes=stock_volumes, table_name=table_name)
    


@app.route('/nasdaq/<symbol>')
def nasdaq(symbol):
    if symbol == 'nasdaq':
        table_name = 'NASDAQ'
        stock_symbol = '^IXIC'
    else:
        return "Invalid stock symbol."

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT date, close, volume FROM { table_name }')
    data = cursor.fetchall()
    conn.close()

    stock_dates = [row[0] for row in data]
    stock_prices = [row[1] for row in data]
    stock_volumes = [row[2] for row in data]

    return render_template('nasdaq.html', stock_symbol=stock_symbol,
                           stock_dates=stock_dates, stock_prices=stock_prices,
                           stock_volumes=stock_volumes, table_name=table_name)


@app.route('/stocks/<stocks>')
def stocks(stocks):
    if stocks == 'msft':
        table_name = 'MSFT'
        stock_symbol = 'MSFT'
    else:
        return "Invalid stock symbol."

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT date, close, volume FROM { table_name }')
    data = cursor.fetchall()
    conn.close()

    stock_dates = [row[0] for row in data]
    stock_prices = [row[1] for row in data]
    stock_volumes = [row[2] for row in data]

    return render_template('stocks.html', stock_symbol=stock_symbol,
                           stock_dates=stock_dates, stock_prices=stock_prices,
                           stock_volumes=stock_volumes, table_name=table_name)
                           




if __name__ == '__main__':
    app.run(debug=True)
