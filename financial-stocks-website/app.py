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
    table_name = 'stocks'  # Table name is always 'stocks'

    if stocks == 'aapl':
        stock_symbol = 'aapl'
    elif stocks == 'msft':
        stock_symbol = 'msft'
    elif stocks == 'amzn':
        stock_symbol = 'amzn'
    elif stocks == 'googl':
        stock_symbol = 'googl'
    elif stocks == 'meta':
        stock_symbol = 'meta'
    elif stocks == 'fb':
        stock_symbol = 'fb'
    elif stocks == 'jpm':
        stock_symbol = 'jpm'
    elif stocks == 'tsla':
        stock_symbol = 'tsla'
    elif stocks == 'brk':
        stock_symbol = 'brk'
    elif stocks == 'nvda':
        stock_symbol = 'nvda'
    elif stocks == 'hd':
        stock_symbol = 'hd'
    elif stocks == 'pypl':
        stock_symbol = 'pypl'
    elif stocks == 'v':
        stock_symbol = 'v'
    elif stocks == 'pg':
        stock_symbol = 'pg'
    elif stocks == 'ma':
        stock_symbol = 'ma'
    elif stocks == 'crm':
        stock_symbol = 'crm'
    elif stocks == 'abt':
        stock_symbol = 'abt'
    elif stocks == 'wmt':
        stock_symbol = 'wmt'
    elif stocks == 'dis':
        stock_symbol = 'dis'
    elif stocks == 'unh':
        stock_symbol = 'unh'
    else:
        return "Invalid stock symbol."

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT date, close, volume FROM {} WHERE ticker = ?'.format(table_name), (stock_symbol,))
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
