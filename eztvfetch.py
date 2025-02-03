import pandas as pd
from datetime import datetime, date

def get_exchange_suffix(ticker):
    """Get the appropriate exchange suffix for the ticker"""
    if ticker in ["SPY", "QQQ", "IWM"]:
        return f"{ticker}:AMEX"
    return ticker  # For other tickers, return as-is

def fetch_options_with_params(ticker):
    """Fetch options data with proper parameters"""
    import eztvscrape
    
    # Format date as YYYYMMDD
    current_date = date.today()
    date_str = current_date.strftime("%Y%m%d")
    
    # Format ticker with exchange
    formatted_ticker = get_exchange_suffix(ticker)
    
    try:
        # Call with correct parameter order: ticker, exchange, date
        return eztvscrape.fetch_options(formatted_ticker, formatted_ticker, date_str)
    except Exception as e:
        raise Exception(f"Error fetching options data: {str(e)}")

def parse_option_data(raw_data):
    """Parse the raw JSON response from eztvscrape into DataFrame format"""
    if not raw_data or 'symbols' not in raw_data:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create mapping of field names to indices
    field_map = {field: idx for idx, field in enumerate(raw_data['fields'])}
    
    calls_data = []
    puts_data = []
    
    for symbol in raw_data['symbols']:
        option_data = {
            'contractSymbol': symbol['s'],
            'strike': symbol['f'][field_map['strike']],
            'lastPrice': (symbol['f'][field_map['bid']] + symbol['f'][field_map['ask']]) / 2,
            'bid': symbol['f'][field_map['bid']],
            'ask': symbol['f'][field_map['ask']],
            'impliedVolatility': symbol['f'][field_map['iv']],
            'delta': symbol['f'][field_map['delta']],
            'gamma': symbol['f'][field_map['gamma']],
            'vega': symbol['f'][field_map['vega']],
            'theta': symbol['f'][field_map['theta']],
            'rho': symbol['f'][field_map['rho']],
            'volume': 0,  # Not provided in eztvscrape data
            'openInterest': 0,  # Not provided in eztvscrape data
            'expiration': str(symbol['f'][field_map['expiration']])
        }
        
        if symbol['f'][field_map['option-type']] == 'call':
            calls_data.append(option_data)
        else:
            puts_data.append(option_data)
    
    calls_df = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
    puts_df = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()
    
    # Add extracted_expiry column
    for df in [calls_df, puts_df]:
        if not df.empty:
            df['extracted_expiry'] = df['expiration'].apply(
                lambda x: datetime.strptime(str(x), '%Y%m%d').date()
            )
    
    return calls_df, puts_df

def get_current_price(options_data):
    """Estimate current price from ATM options"""
    if not options_data or 'symbols' not in options_data:
        return None
        
    # Find ATM options and estimate price
    calls_near_money = []
    puts_near_money = []
    
    field_map = {field: idx for idx, field in enumerate(options_data['fields'])}
    
    for symbol in options_data['symbols']:
        strike = symbol['f'][field_map['strike']]
        theo_price = symbol['f'][field_map['theoPrice']]
        
        if symbol['f'][field_map['option-type']] == 'call':
            calls_near_money.append((strike, theo_price))
        else:
            puts_near_money.append((strike, theo_price))
            
    if not calls_near_money or not puts_near_money:
        return None
        
    # Get the most ATM call and put
    calls_near_money.sort(key=lambda x: abs(x[0] - x[1]))
    puts_near_money.sort(key=lambda x: abs(x[0] - x[1]))
    
    # Average the implied prices from the most ATM call and put
    call_implied = calls_near_money[0][0] + calls_near_money[0][1]
    put_implied = puts_near_money[0][0] - puts_near_money[0][1]
    
    return (call_implied + put_implied) / 2
