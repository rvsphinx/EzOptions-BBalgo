import requests
import json

def fetch_options(ticker, exchange, date):
    url = "https://scanner.tradingview.com/options/scan2?label-product=options-builder"
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "text/plain;charset=UTF-8",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\", \"Google Chrome\";v=\"132\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Referer": "https://www.tradingview.com/",
        "Referrer-Policy": "origin-when-cross-origin"
    }
    payload = {
        "columns": ["ask", "bid", "currency", "delta", "expiration", "gamma", "iv", "option-type", "pricescale", "rho", "root", "strike", "theoPrice", "theta", "vega"],
        "filter": [
            {"left": "type", "operation": "equal", "right": "option"},
            {"left": "expiration", "operation": "equal", "right": int(date)},
            {"left": "root", "operation": "equal", "right" :ticker}
        ],
        "ignore_unknown_fields": False,
        "index_filters": [
            {"name": "underlying_symbol", "values": [f"{exchange}:{ticker}"]}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()

    # Check if options are returned, if not, try adding "W" to the filter ticker
    if not data.get('data'):
        payload['filter'][2]['right'] = f"{ticker}W"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()

        # Debugging: Print the modified payload and response
        print("Modified payload sent to API:")
        print(json.dumps(payload, indent=4))
        print("Response from API after modification:")
        print(json.dumps(data, indent=4))

    return data

def fetch_price(ticker):
    url = "https://scanner.tradingview.com/global/scan2?label-product=options-builder"
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "text/plain;charset=UTF-8",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\", \"Google Chrome\";v=\"132\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Referer": "https://www.tradingview.com/",
        "Referrer-Policy": "origin-when-cross-origin"
    }
    payload = {
        "columns": ["close"],
        "ignore_unknown_fields": False,
        "symbols": {
            "tickers": [ticker]
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()

    # Extract the price from the response
    price = data['symbols'][0]['f'][0] if data['totalCount'] > 0 else None

    return price
