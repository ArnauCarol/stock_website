import yfinance as yf
import pandas as pd

# List of related ticker pairs
related_ticker_pairs = [
    ['^GSPC', '^VIX'],        # S&P 500 and VIX
    ['^DJI', '^VIX'],         # Dow Jones and VIX
    ['^IXIC', '^VIX'],        # Nasdaq and VIX
    ['EURUSD=X', 'USDJPY=X'], # EUR/USD and USD/JPY
    ['AUDUSD=X', 'USDJPY=X'], # AUD/USD and USD/JPY
    ['BZ=F', 'GC=F'],         # Brent Crude and Gold
    ['BZ=F', 'SI=F'],         # Brent Crude and Silver
    ['GC=F', 'SI=F'],         # Gold and Silver
    ['PL=F', 'PA=F'],         # Platinum and Palladium
    ['^TNX', '^VIX'],         # 10-Year Treasury Yield and VIX
    ['META', 'AAPL'],           # Facebook and Apple
    ['AAPL', 'AMZN'],         # Apple and Amazon
    ['AMZN', 'NFLX'],         # Amazon and Netflix
    ['GOOGL', 'AAPL'],        # Google and Apple
    ['JPY=X', '^VIX'],        # Japanese Yen and VIX
    ['VNQ', '^TNX'],          # Real Estate ETF and 10-Year Treasury Yield
]

ticker_to_company = {
    '^GSPC': 'S&P 500',
    '^VIX': 'VIX',
    '^DJI': 'Dow Jones',
    '^IXIC': 'Nasdaq',
    'EURUSD=X': 'EUR/USD',
    'USDJPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD',
    'BZ=F': 'Brent Crude',
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'PL=F': 'Platinum',
    'PA=F': 'Palladium',
    '^TNX': '10-Year Treasury Yield',
    'META': 'Facebook',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'NFLX': 'Netflix',
    'GOOGL': 'Google',
    'JPY=X': 'Japanese Yen',
    'VNQ': 'VNQ',
}

start_date = '2020-01-01'
end_date = '2023-01-01'

# Create a DataFrame to store correlation coefficients
correlation_data = []

# Iterate over each pair of tickers and calculate correlation
for pair in related_ticker_pairs:
    ticker1 = yf.download(pair[0], start=start_date, end=end_date)['Adj Close']
    ticker2 = yf.download(pair[1], start=start_date, end=end_date)['Adj Close']
    correlation = ticker1.corr(ticker2)
    company1 = ticker_to_company[pair[0]]
    company2 = ticker_to_company[pair[1]]
    correlation_data.append([company1, company2, correlation])

# Create a DataFrame from the correlation data
correlation_df = pd.DataFrame(correlation_data, columns=['Company 1', 'Company 2', 'Correlation'])

# Display the correlation DataFrame
print(correlation_df)

correlation_table_html = correlation_df.to_html(
    index=False,
    classes='table table-bordered table-hover'
)

# Your HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial Stocks Website</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }}
        header {{
            background-color: #222;
            color: #fff;
            text-align: center;
            padding: 2em;
        }}
        header ul {{
            list-style-type: none;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: flex-end;
        }}
        header ul li {{
            margin-right: 70px;
        }}
        header ul li a {{
            color: #fff;
            text-decoration: none;
        }}
        .dropdown {{
            position: relative;
            display: inline-block;
        }}
        .dropdown-content {{
            display: none;
            position: absolute;
            background-color: #f9f9f9;
            min-width: 160px;
            z-index: 1;
        }}
        .dropdown-content a {{
            color: black;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
        }}
        .dropdown-content a:hover {{
            background-color: #f1f1f1;
        }}
        .dropdown:hover .dropdown-content {{
            display: block;
        }}
        .content {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h2 {{
            margin-top: 0;
            color: #333;
        }}
        p {{
            margin-bottom: 16px;
            line-height: 1.6;
            color: #555;
        }}
        ol {{
            margin-left: 20px;
        }}
        strong {{
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <header>
        <h1>STOCK.AI</h1>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li class="dropdown">
                    <a href="javascript:void(0)">Stocks</a>
                    <div class="dropdown-content">
                        <a href="/sp500/sp500">S&P500</a>
                        <a href="/ibex35/ibex35">IBEX35</a>
                        <a href="/nasdq/nasdq">NASDQ</a>
                    </div>
                </li>
                <li><a href="/">Currency</a></li>
            </ul>
        </nav>
    </header>
    <section class="content">
        <h2>Why Use AI?</h2>
        <p>
            Certainly! Here's a list of various financial instruments and markets that are related and often exhibit some form of interaction or correlation with each other:
            
            <!-- Embed the correlation table HTML here -->
            {correlation_table_html}
        </p>
        <!-- ... Rest of your content ... -->
    </section>
</body>
</html>
"""



# Replace the placeholder with the correlation table HTML
html_template = html_template.format(correlation_table_html=correlation_table_html)

# Write the HTML table to an HTML file
with open('templates/rel_stocks.html', 'w') as f:
    f.write(html_template)