import sqlite3
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stocks/<symbol>')
def stocks(symbol):
    if symbol == 'sp500':
        table_name = 'SP500'
        stock_symbol = '^GSPC'
    elif symbol == 'ibex35':
        table_name = 'IBEX35'
        stock_symbol = '^IBEX'
    elif symbol == 'nasdq':
        table_name = 'NASDQ'
        stock_symbol = 'NQ=F'
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

    return render_template('stocks.html', stock_symbol=stock_symbol,
                           stock_dates=stock_dates, stock_prices=stock_prices,
                           stock_volumes=stock_volumes, table_name=table_name)

if __name__ == '__main__':
    app.run(debug=True)
