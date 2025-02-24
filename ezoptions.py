import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
from math import log, sqrt
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
import re
import time
from scipy.stats import norm
import requests
import json
import base64
import threading
from random import randint
from contextlib import contextmanager
from scipy.interpolate import griddata
import numpy as np

# Add thread context management
@contextmanager
def st_thread_context():
    """Thread context management for Streamlit"""
    try:
        if not hasattr(threading.current_thread(), '_StreamlitThread__cached_st'):
           
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*missing ScriptRunContext.*')
        yield
    finally:
        pass


with st_thread_context():
    st.set_page_config(layout="wide")

# Initialize session state for colors if not already set
if 'call_color' not in st.session_state:
    st.session_state.call_color = '#00FF00'  # Default green for calls
if 'put_color' not in st.session_state:
    st.session_state.put_color = '#FF0000'   # Default red for puts
if 'vix_color' not in st.session_state:
    st.session_state.vix_color = '#800080'   # Default purple for VIX

# -------------------------------
# Helper Functions
# -------------------------------
def format_ticker(ticker):
    """Helper function to format tickers for indices"""
    ticker = ticker.upper()
    if ticker == "SPX":
        return "^SPX"
    elif ticker == "NDX":
        return "^NDX"
    elif ticker == "VIX":
        return "^VIX"
    elif ticker == "DJI":
        return "^DJI"
    elif ticker == "RUT":
        return "^RUT"
    return ticker

def fetch_options_for_date(ticker, date, S=None):
    """
    Fetches option chains for the given ticker and date.
    Returns two DataFrames: one for calls and one for puts.
    """
    print(f"Fetching option chain for {ticker} EXP {date}")
    stock = yf.Ticker(ticker)
    try:
        if S is None:
            S = get_current_price(ticker)
        chain = stock.option_chain(date)
        calls = chain.calls
        puts = chain.puts
        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options chain for date {date}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def clear_page_state():
    """Clear all page-specific content and containers"""
    for key in list(st.session_state.keys()):
        if key.startswith(('container_', 'chart_', 'table_', 'page_')):
            del st.session_state[key]
    
    if 'current_page_container' in st.session_state:
        del st.session_state['current_page_container']
    
    st.empty()

def extract_expiry_from_contract(contract_symbol):
    """
    Extracts the expiration date from an option contract symbol.
    Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date formats.
    """
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                # Parse as YYMMDD
                expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            else:
                # Parse as YYYYMMDD
                expiry_date = datetime.strptime(date_str, "%Y%m%d").date()
            return expiry_date
        except ValueError:
            return None
    return None

def add_current_price_line(fig, current_price):
    """
    Adds a vertical dashed white line at the current price to a Plotly figure.
    """
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"{current_price}",
        annotation_position="top",
    )
    return fig

def get_screener_data(screener_type):
    """Fetch screener data from Yahoo Finance"""
    try:
        response = yf.screen(screener_type)
        if isinstance(response, dict) and 'quotes' in response:
            data = []
            for quote in response['quotes']:
                # Extract relevant information
                info = {
                    'symbol': quote.get('symbol', ''),
                    'shortName': quote.get('shortName', ''),
                    'regularMarketPrice': quote.get('regularMarketPrice', 0),
                    'regularMarketChangePercent': quote.get('regularMarketChangePercent', 0),
                    'regularMarketVolume': quote.get('regularMarketVolume', 0),
                }
                data.append(info)
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching screener data: {e}")
        return pd.DataFrame()

# -------------------------------
# Fetch all options experations and add extract expiry
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
    print(f"Fetching avaiable expirations for {ticker}")  # Add print statement
    stock = yf.Ticker(ticker)
    all_calls = []
    all_puts = []
    
    if stock.options:
        # Get current market date
        current_market_date = datetime.now().date()
        
        for exp in stock.options:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                
                # Only process options that haven't expired
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                if exp_date >= current_market_date:
                    if not calls.empty:
                        calls = calls.copy()
                        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                        all_calls.append(calls)
                    if not puts.empty:
                        puts = puts.copy()
                        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                        all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching chain for expiry {exp}: {e}")
                continue
    else:
        try:
            # Get next valid expiration
            next_exp = stock.options[0] if stock.options else None
            if next_exp:
                chain = stock.option_chain(next_exp)
                calls = chain.calls
                puts = chain.puts
                if not calls.empty:
                    calls = calls.copy()
                    calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                    all_calls.append(calls)
                if not puts.empty:
                    puts = puts.copy()
                    puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                    all_puts.append(puts)
        except Exception as e:
            st.error(f"Error fetching fallback options data: {e}")
    
    if all_calls:
        combined_calls = pd.concat(all_calls, ignore_index=True)
    else:
        combined_calls = pd.DataFrame()
    if all_puts:
        combined_puts = pd.concat(all_puts, ignore_index=True)
    else:
        combined_puts = pd.DataFrame()
    
    return combined_calls, combined_puts

# Charts and price fetching
def get_current_price(ticker):
    """Get current price with fallback logic"""
    print(f"Fetching current price for {ticker}")
    formatted_ticker = ticker.replace('%5E', '^')
    
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                return round(float(price), 2)
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
    
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice")
        if price is None:
            price = stock.fast_info.get("lastPrice")
        if price is not None:
            return round(float(price), 2)
    except Exception as e:
        print(f"Yahoo Finance error: {str(e)}")
    
    return None

def create_oi_volume_charts(calls, puts, S):
    # Remove get_current_price call since S is passed in
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    # Calculate strike range around current price
    min_strike = S - st.session_state.strike_range
    max_strike = S + st.session_state.strike_range
    
    # Filter data based on strike range
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    calls_df = calls[['strike', 'openInterest', 'volume']].copy()
    calls_df['OptionType'] = 'Call'
    
    puts_df = puts[['strike', 'openInterest', 'volume']].copy()
    puts_df['OptionType'] = 'Put'
    
    combined = pd.concat([calls_df, puts_df], ignore_index=True)
    combined.sort_values(by='strike', inplace=True)
    
    # Calculate Net Open Interest and Net Volume using filtered data
    net_oi = calls.groupby('strike')['openInterest'].sum() - puts.groupby('strike')['openInterest'].sum()
    net_volume = calls.groupby('strike')['volume'].sum() - puts.groupby('strike')['volume'].sum()
    
    # Add padding for x-axis range
    padding = st.session_state.strike_range * 0.1
    
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    fig_oi = px.bar(
        combined,
        x='strike',
        y='openInterest',
        color='OptionType',
        title='Open Interest by Strike',
        barmode='group',
        color_discrete_map={'Call': call_color, 'Put': put_color}
    )
    
    # Add Net OI trace as bar
    if st.session_state.show_net:
        fig_oi.add_trace(go.Bar(
            x=net_oi.index, 
            y=net_oi.values, 
            name='Net OI', 
            marker=dict(color=[call_color if val >= 0 else put_color for val in net_oi.values])
        ))
    
    # Update OI chart layout with text size settings
    fig_oi.update_layout(
        title=dict(
            text='Open Interest by Strike',
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Open Interest',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10),
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_volume = px.bar(
        combined,
        x='strike',
        y='volume',
        color='OptionType',
        title='Volume by Strike',
        barmode='group',
        color_discrete_map={'Call': call_color, 'Put': put_color}
    )
    
    # Add Net Volume trace as bar
    if st.session_state.show_net:
        fig_volume.add_trace(go.Bar(
            x=net_volume.index, 
            y=net_volume.values, 
            name='Net Volume', 
            marker=dict(color=[call_color if val >= 0 else put_color for val in net_volume.values])
        ))
    
    # Update Volume chart layout with text size settings
    fig_volume.update_layout(
        title=dict(
            text='Volume by Strike',
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10),
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_oi.update_xaxes(rangeslider=dict(visible=True))
    fig_volume.update_xaxes(rangeslider=dict(visible=True))
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    # Apply show/hide settings for calls and puts
    if not st.session_state.show_calls:
        fig_oi.for_each_trace(lambda trace: trace.update(visible='legendonly') 
                             if trace.name == 'Call' else None)
        fig_volume.for_each_trace(lambda trace: trace.update(visible='legendonly') 
                                if trace.name == 'Call' else None)
    
    if not st.session_state.show_puts:
        fig_oi.for_each_trace(lambda trace: trace.update(visible='legendonly') 
                             if trace.name == 'Put' else None)
        fig_volume.for_each_trace(lambda trace: trace.update(visible='legendonly') 
                                if trace.name == 'Put' else None)
    
    return fig_oi, fig_volume

def create_donut_chart(call_volume, put_volume):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    # Get colors directly from session state at creation time
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        title_font_size=st.session_state.chart_text_size + 8,  # Title slightly larger
        showlegend=True,
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        )
    )
    fig.update_traces(
        hoverinfo='label+percent+value',
        marker=dict(colors=[call_color, put_color]),
        textfont=dict(size=st.session_state.chart_text_size)
    )
    return fig

# Greek Calculations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_risk_free_rate():
    """Fetch the current risk-free rate from the 3-month Treasury Bill yield with caching."""
    try:
        # Get current price for the 3-month Treasury Bill
        irx_rate = get_current_price("^IRX")
        
        if irx_rate is not None:
            # Convert percentage to decimal (e.g., 5.2% to 0.052)
            risk_free_rate = irx_rate / 100
        else:
            # Fallback to a default value if price fetch fails
            risk_free_rate = 0.02  # 2% as fallback
            print("Using fallback risk-free rate of 2%")
            
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.02  # 2% as fallback

# Initialize risk-free rate in session state if not already present
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = get_risk_free_rate()

def calculate_greeks(flag, S, K, t, sigma):
    """
    Calculate delta, gamma and vanna for an option.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Update py_vollib calls to use the risk-free rate
        delta_val = bs_delta(flag, S, K, t, r, sigma)
        gamma_val = bs_gamma(flag, S, K, t, r, sigma)
        vega_val = bs_vega(flag, S, K, t, r, sigma)
        
        # Correct vanna calculation with risk-free rate
        vanna_val = -norm.pdf(d1) * d2 / sigma
        
        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

def calculate_charm(flag, S, K, t, sigma):
    """
    Calculate charm (dDelta/dTime) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        norm_d1 = norm.pdf(d1)
        
        # Correct charm formula with risk-free rate
        if flag == 'c':
            charm = -norm_d1 * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
        else:  # put
            charm = -norm_d1 * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
            charm = -charm  # Negative for puts
        
        return charm
    except Exception as e:
        st.error(f"Error calculating charm: {e}")
        return None

def calculate_speed(flag, S, K, t, sigma):
    """
    Calculate speed (dGamma/dSpot) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        
        # Correct speed formula with risk-free rate
        gamma = bs_gamma(flag, S, K, t, r, sigma)
        speed = -gamma * (d1/(sigma * sqrt(t)) + 1) / S
        
        return speed
    except Exception as e:
        st.error(f"Error calculating speed: {e}")
        return None

def calculate_vomma(flag, S, K, t, sigma):
    """
    Calculate vomma (dVega/dVol) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Correct vomma formula with risk-free rate
        vega = bs_vega(flag, S, K, t, r, sigma)
        vomma = vega * (d1 * d2) / sigma
        
        return vomma
    except Exception as e:
        st.error(f"Error calculating vomma: {e}")
        return None

# Add error handling for fetching the last price to avoid KeyError.
def get_last_price(stock):
    """Helper function to get the last price of the stock."""
    return get_current_price(stock.ticker)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = datetime.now().date()
        # Allow expirations within -1 days
        days_difference = (expiry_date - current_market_date).days
        return days_difference >= -1
    except Exception:
        return False

def is_valid_trading_day(expiry_date, current_date):
    """Helper function to check if expiry is within valid trading window"""
    days_difference = (expiry_date - current_date).days
    return days_difference >= -1

def fetch_and_process_multiple_dates(ticker, expiry_dates, process_func):
    """
    Fetches and processes data for multiple expiration dates.
    
    Args:
        ticker: Stock ticker symbol
        expiry_dates: List of expiration dates
        process_func: Function to process data for each date
        
    Returns:
        Tuple of processed calls and puts DataFrames
    """
    all_calls = []
    all_puts = []
    
    for date in expiry_dates:
        result = process_func(ticker, date)
        if result is not None:
            calls, puts = result
            if not calls.empty:
                calls['expiry_date'] = date  # Add expiry date column
                all_calls.append(calls)
            if not puts.empty:
                puts['expiry_date'] = date  # Add expiry date column
                all_puts.append(puts)
    
    if all_calls and all_puts:
        combined_calls = pd.concat(all_calls, ignore_index=True)
        combined_puts = pd.concat(all_puts, ignore_index=True)
        return combined_calls, combined_puts
    return pd.DataFrame(), pd.DataFrame()

def get_combined_intraday_data(ticker):
    """Get intraday data with fallback logic"""
    formatted_ticker = ticker.replace('%5E', '^')
    stock = yf.Ticker(ticker)
    intraday_data = stock.history(period="1d", interval="1m")
    
    # Filter for market hours (9:30 AM to 4:00 PM ET)
    if not intraday_data.empty:
        intraday_data.index = intraday_data.index.tz_localize(None)
        market_start = pd.Timestamp(intraday_data.index[0].date()).replace(hour=9, minute=30)
        market_end = pd.Timestamp(intraday_data.index[0].date()).replace(hour=16, minute=0)
        intraday_data = intraday_data.between_time('9:30', '16:00')
    
    if intraday_data.empty:
        return None, None, None
    
    # Get VIX data if overlay is enabled
    vix_data = None
    if st.session_state.show_vix_overlay:
        vix = yf.Ticker('^VIX')
        vix_intraday = vix.history(period="1d", interval="1m")
        if not vix_intraday.empty:
            vix_intraday.index = vix_intraday.index.tz_localize(None)
            vix_data = vix_intraday.between_time('9:30', '16:00')
    
    intraday_data = intraday_data.copy()
    yahoo_last_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else None
    latest_price = yahoo_last_price
    
    # Use ^GSPC for SPX
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                latest_price = round(float(price), 2)
                last_idx = intraday_data.index[-1]
                intraday_data.loc[last_idx, 'Close'] = latest_price
                intraday_data.loc[last_idx, 'Open'] = latest_price
                intraday_data.loc[last_idx, 'High'] = max(latest_price, intraday_data.loc[last_idx, 'High'])
                intraday_data.loc[last_idx, 'Low'] = min(latest_price, intraday_data.loc[last_idx, 'Low'])
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
            latest_price = yahoo_last_price
    
    return intraday_data, latest_price, vix_data

def create_iv_surface(calls_df, puts_df, current_price, selected_dates=None):
    """Create data for IV surface plot with enhanced smoothing and data validation."""
    # Filter by selected dates if provided
    if selected_dates:
        calls_df = calls_df[calls_df['extracted_expiry'].isin(selected_dates)]
        puts_df = puts_df[puts_df['extracted_expiry'].isin(selected_dates)]
    
    # Combine calls and puts and drop rows with NaN values
    options_data = pd.concat([calls_df, puts_df])
    options_data = options_data.dropna(subset=['impliedVolatility', 'strike', 'extracted_expiry'])
    
    if options_data.empty:
        st.warning("No valid options data available for IV surface.")
        return None, None, None
    
    # Calculate moneyness and months to expiration
    options_data['moneyness'] = options_data['strike'].apply(
        lambda x: (x / current_price) * 100
    )
    
    options_data['months'] = options_data['extracted_expiry'].apply(
        lambda x: (x - datetime.now().date()).days / 30.44
    )
    
    # Remove extreme values
    for col in ['impliedVolatility', 'moneyness', 'months']:
        q1 = options_data[col].quantile(0.01)
        q99 = options_data[col].quantile(0.99)
        options_data = options_data[
            (options_data[col] >= q1) & 
            (options_data[col] <= q99)
        ]
    
    if options_data.empty:
        st.warning("No valid data points after filtering.")
        return None, None, None
    
    # Create grid for interpolation
    moneyness_range = np.linspace(85, 115, 200)
    months_range = np.linspace(
        options_data['months'].min(),
        options_data['months'].max(),
        200
    )
    
    # Create meshgrid
    X, Y = np.meshgrid(moneyness_range, months_range)
    
    try:
        # Prepare data for interpolation
        points = options_data[['moneyness', 'months']].values
        values = options_data['impliedVolatility'].values * 100
        
        # Initial interpolation
        Z = griddata(
            points,
            values,
            (X, Y),
            method='linear',  # Start with linear interpolation
            fill_value=np.nan
        )
        
        # Fill remaining NaN values with nearest neighbor interpolation
        mask = np.isnan(Z)
        Z[mask] = griddata(
            points,
            values,
            (X[mask], Y[mask]),
            method='nearest'
        )
        
        # Apply Gaussian smoothing with multiple passes
        if not np.isnan(Z).any():  # Only smooth if we have valid data
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=1.5)
            Z = gaussian_filter(Z, sigma=0.75)
            Z = gaussian_filter(Z, sigma=0.5)
        
        return X, Y, Z
        
    except Exception as e:
        st.error(f"Error creating IV surface: {str(e)}")
        return None, None, None

#Streamlit UI
st.title("Ez Options Stock Data")

# Modify the reset_session_state function to preserve color settings
def reset_session_state():
    """Reset all session state variables except for essential ones"""
    # Keep track of keys we want to preserve
    preserved_keys = {
        'current_page', 
        'initialized', 
        'saved_ticker', 
        'call_color', 
        'put_color',
        'show_calls', 
        'show_puts',
        'show_net',
        'strike_range',
        'chart_type',
        'refresh_rate',
        'intraday_chart_type',
        'candlestick_type' 
    }
    
    # Initialize visibility settings if they don't exist
    if 'show_calls' not in st.session_state:
        st.session_state.show_calls = True
    if 'show_puts' not in st.session_state:
        st.session_state.show_puts = True
    if 'show_net' not in st.session_state:
        st.session_state.show_net = True
    
    preserved_values = {key: st.session_state[key] 
                       for key in preserved_keys 
                       if key in st.session_state}
    
    # Clear everything safely
    for key in list(st.session_state.keys()):
        if key not in preserved_keys:
            try:
                del st.session_state[key]
            except KeyError:
                pass
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value

    # Reset expiry selection keys explicitly
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'max_pain_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add near the top with other session state initializations
if 'selected_expiries' not in st.session_state:
    st.session_state.selected_expiries = {}

@st.fragment
def expiry_selector_fragment(page_name, available_dates):
    """Fragment for expiry date selection that properly resets"""
    container = st.empty()
    
    # Initialize session state for this page's selections
    state_key = f"{page_name}_selected_dates"
    widget_key = f"{page_name}_expiry_selector"
    
    # Initialize previous selection state if not exists
    if f"{widget_key}_prev" not in st.session_state:
        st.session_state[f"{widget_key}_prev"] = []
    
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    
    with container:
        selected = st.multiselect(
            "Select Expiration Date(s) (max 14):",
            options=available_dates,
            default=st.session_state[state_key],
            max_selections=14,
            key=widget_key
        )
        
        # Check if selection changed
        if selected != st.session_state[f"{widget_key}_prev"]:
            st.session_state[state_key] = selected
            st.session_state[f"{widget_key}_prev"] = selected.copy()
            if selected:  # Only rerun if there are selections
                st.rerun()
    
    return selected, container

def handle_page_change(new_page):
    """Handle page navigation and state management"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = new_page
        return True
    
    if st.session_state.current_page != new_page:
        # Clear page-specific state
        old_state_key = f"{st.session_state.current_page}_selected_dates"
        if old_state_key in st.session_state:
            del st.session_state[old_state_key]
            
        if 'expiry_selector_container' in st.session_state:
            st.session_state.expiry_selector_container.empty()
        
        st.session_state.current_page = new_page
        reset_session_state()
        st.rerun()
        return True
    
    return False

# Save selected ticker
def save_ticker(ticker):
    st.session_state.saved_ticker = ticker

st.sidebar.title("Navigation")
pages = ["Dashboard", "Volume Ratio", "OI & Volume", "Gamma Exposure", "Delta Exposure", 
          "Vanna Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Delta-Adjusted Value Index", "Max Pain", "GEX Surface", "IV Surface",
          "Analysis", "Calculated Greeks"]

# Track the previous page in session state
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None

new_page = st.sidebar.radio("Select a page:", pages)

# Check if the page has changed
if st.session_state.previous_page != new_page:
    st.session_state.previous_page = new_page
    handle_page_change(new_page)
    # Clear out any page-specific expiry selections
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'max_pain_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add after st.sidebar.title("Navigation")
def chart_settings():
    with st.sidebar.expander("Chart Settings", expanded=False):
        st.write("Colors:")
        new_call_color = st.color_picker("Calls", st.session_state.call_color)
        new_put_color = st.color_picker("Puts", st.session_state.put_color)
        
        # Add intraday chart type selection
        if 'intraday_chart_type' not in st.session_state:
            st.session_state.intraday_chart_type = 'Candlestick'
        
        if 'candlestick_type' not in st.session_state:
            st.session_state.candlestick_type = 'Filled'
        
        intraday_type = st.selectbox(
            "Intraday Chart Type:",
            options=['Candlestick', 'Line'],
            index=['Candlestick', 'Line'].index(st.session_state.intraday_chart_type)
        )
        
        # Only show candlestick type selection when candlestick chart is selected
        if intraday_type == 'Candlestick':
            candlestick_type = st.selectbox(
                "Candlestick Style:",
                options=['Filled', 'Hollow'],
                index=['Filled', 'Hollow'].index(st.session_state.candlestick_type)
            )
            
            if candlestick_type != st.session_state.candlestick_type:
                st.session_state.candlestick_type = candlestick_type
        
        # Update session state when intraday chart type changes
        if intraday_type != st.session_state.intraday_chart_type:
            st.session_state.intraday_chart_type = intraday_type

        if 'show_vix_overlay' not in st.session_state:
            st.session_state.show_vix_overlay = False
        
        # Group VIX settings together
        st.write("VIX Settings:")
        show_vix = st.checkbox("VIX Overlay", value=st.session_state.show_vix_overlay)
        if show_vix:
            new_vix_color = st.color_picker("VIX Color", st.session_state.vix_color)
            if new_vix_color != st.session_state.vix_color:
                st.session_state.vix_color = new_vix_color
        
        if show_vix != st.session_state.show_vix_overlay:
            st.session_state.show_vix_overlay = show_vix

        if 'chart_text_size' not in st.session_state:
            st.session_state.chart_text_size = 12  # Default text size
            
        new_text_size = st.number_input(
            "Chart Text Size",
            min_value=10,
            max_value=30,
            value=st.session_state.chart_text_size,
            step=1,
            help="Adjust the size of text in charts (titles, labels, etc.)"
        )
        
        # Update session state and trigger rerun if text size changes
        if new_text_size != st.session_state.chart_text_size:
            st.session_state.chart_text_size = new_text_size
        
        # Update session state and trigger rerun if either color changes
        if new_call_color != st.session_state.call_color or new_put_color != st.session_state.put_color:
            st.session_state.call_color = new_call_color
            st.session_state.put_color = new_put_color

        st.write("Show/Hide Elements:")
        # Initialize visibility settings if not already set
        if 'show_calls' not in st.session_state:
            st.session_state.show_calls = False
        if 'show_puts' not in st.session_state:
            st.session_state.show_puts = False
        if 'show_net' not in st.session_state:
            st.session_state.show_net = True

        # Visibility toggles
        show_calls = st.checkbox("Show Calls", value=st.session_state.show_calls)
        show_puts = st.checkbox("Show Puts", value=st.session_state.show_puts)
        show_net = st.checkbox("Show Net", value=st.session_state.show_net)

        # Update session state when visibility changes
        if show_calls != st.session_state.show_calls or show_puts != st.session_state.show_puts or show_net != st.session_state.show_net:
            st.session_state.show_calls = show_calls
            st.session_state.show_puts = show_puts
            st.session_state.show_net = show_net

        # Initialize strike range in session state
        if 'strike_range' not in st.session_state:
            st.session_state.strike_range = 20.0
        
        # Add strike range control
        st.session_state.strike_range = st.number_input(
            "Strike Range (±)",
            min_value=1.0,
            max_value=2000.0,
            value=st.session_state.strike_range,
            step=1.0,
            key="strike_range_sidebar"
        )

        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = 'Bar'  # Default chart type

        chart_type = st.selectbox(
            "Chart Type:",
            options=['Bar', 'Scatter', 'Line', 'Area'],
            index=['Bar', 'Scatter', 'Line', 'Area'].index(st.session_state.chart_type)
        )

        # Update session state when chart type changes
        if chart_type != st.session_state.chart_type:
            st.session_state.chart_type = chart_type

        # Add refresh rate control before chart type
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = 10  # Default refresh rate
        
        new_refresh_rate = st.number_input(
            "Auto-Refresh Rate (seconds)",
            min_value=10,
            max_value=300,
            value=int(st.session_state.refresh_rate),
            step=1,
            help="How often to auto-refresh the page (minimum 10 seconds)"
        )
        
        if new_refresh_rate != st.session_state.refresh_rate:
            print(f"Changing refresh rate from {st.session_state.refresh_rate} to {new_refresh_rate} seconds")
            st.session_state.refresh_rate = float(new_refresh_rate)
            st.cache_data.clear()
            st.rerun()

        # Add GEX Type selector after chart type
        if 'gex_type' not in st.session_state:
            st.session_state.gex_type = 'Net'  # Default to Net GEX
        
        gex_type = st.selectbox(
            "Gamma Exposure Type:",
            options=['Net', 'Absolute'],
            index=['Net', 'Absolute'].index(st.session_state.gex_type)
        )

        # Update session state when GEX type changes
        if gex_type != st.session_state.gex_type:
            st.session_state.gex_type = gex_type

# Call the regular function instead of the fragment
chart_settings()

# Use the saved ticker and expiry date if available
saved_ticker = st.session_state.get("saved_ticker", "")
saved_expiry_date = st.session_state.get("saved_expiry_date", None)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = datetime.now().date()
        # For future dates, ensure they're treated as valid
        return expiry_date >= current_market_date
    except Exception:
        return False

def compute_greeks_and_charts(ticker, expiry_date_str, page_key, S):
    """Compute greeks and create charts for options data"""
    if not expiry_date_str:
        st.warning("Please select an expiration date.")
        return None, None, None, None, None, None
        
    calls, puts = fetch_options_for_date(ticker, expiry_date_str, S)
    if calls.empty and puts.empty:
        st.warning("No options data available for this ticker.")
        return None, None, None, None, None, None

    combined = pd.concat([calls, puts])
    combined = combined.dropna(subset=['extracted_expiry'])
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    calls = calls[calls['extracted_expiry'] == selected_expiry]
    puts = puts[puts['extracted_expiry'] == selected_expiry]

    # Always use get_current_price to ensure consistent price source
    if S is None:
        st.error("Could not fetch underlying price.")
        return None, None, None, None, None, None

    S = float(S)  # Ensure price is float
    today = datetime.today().date()
    t_days = (selected_expiry - today).days
    if not is_valid_trading_day(selected_expiry, today):
        st.error("The selected expiration date is in the past!")
        return None, None, None, None, None, None

    t = t_days / 365.0

    # Compute Greeks for Gamma, Vanna, Delta, Charm, Speed, and Vomma
    def compute_greeks(row, flag, greek_type):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
            if greek_type == "gamma":
                return gamma_val
            elif greek_type == "vanna":
                return vanna_val
            elif greek_type == "delta":
                return delta_val
        except Exception:
            return None

    def compute_charm(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            charm_val = calculate_charm(flag, S, row["strike"], t, sigma)
            return charm_val
        except Exception:
            return None

    def compute_speed(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            speed_val = calculate_speed(flag, S, row["strike"], t, sigma)
            return speed_val
        except Exception:
            return None

    def compute_vomma(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            vomma_val = calculate_vomma(flag, S, row["strike"], t, sigma)
            return vomma_val
        except Exception:
            return None

    calls = calls.copy()
    puts = puts.copy()
    calls["calc_gamma"] = calls.apply(lambda row: compute_greeks(row, "c", "gamma"), axis=1)
    puts["calc_gamma"] = puts.apply(lambda row: compute_greeks(row, "p", "gamma"), axis=1)
    calls["calc_vanna"] = calls.apply(lambda row: compute_greeks(row, "c", "vanna"), axis=1)
    puts["calc_vanna"] = puts.apply(lambda row: compute_greeks(row, "p", "vanna"), axis=1)
    calls["calc_delta"] = calls.apply(lambda row: compute_greeks(row, "c", "delta"), axis=1)
    puts["calc_delta"] = puts.apply(lambda row: compute_greeks(row, "p", "delta"), axis=1)
    calls["calc_charm"] = calls.apply(lambda row: compute_charm(row, "c"), axis=1)
    puts["calc_charm"] = puts.apply(lambda row: compute_charm(row, "p"), axis=1)
    calls["calc_speed"] = calls.apply(lambda row: compute_speed(row, "c"), axis=1)
    puts["calc_speed"] = puts.apply(lambda row: compute_speed(row, "p"), axis=1)
    calls["calc_vomma"] = calls.apply(lambda row: compute_vomma(row, "c"), axis=1)
    puts["calc_vomma"] = puts.apply(lambda row: compute_vomma(row, "p"), axis=1)

    calls = calls.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma"])
    puts = puts.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma"])

    calls["GEX"] = calls["calc_gamma"] * calls["openInterest"] * 100
    puts["GEX"] = puts["calc_gamma"] * puts["openInterest"] * 100
    calls["VEX"] = calls["calc_vanna"] * calls["openInterest"] * 100
    puts["VEX"] = puts["calc_vanna"] * puts["openInterest"] * 100
    calls["DEX"] = calls["calc_delta"] * calls["openInterest"] * 100
    puts["DEX"] = puts["calc_delta"] * puts["openInterest"] * 100
    calls["Charm"] = calls["calc_charm"] * calls["openInterest"] * 100
    puts["Charm"] = puts["calc_charm"] * puts["openInterest"] * 100
    calls["Speed"] = calls["calc_speed"] * calls["openInterest"] * 100
    puts["Speed"] = puts["calc_speed"] * puts["openInterest"] * 100
    calls["Vomma"] = calls["calc_vomma"] * calls["openInterest"] * 100
    puts["Vomma"] = puts["calc_vomma"] * puts["openInterest"] * 100

    return calls, puts, S, t, selected_expiry, today

def create_exposure_bar_chart(calls, puts, exposure_type, title, S):
    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Filter out zero values
    calls_df = calls[['strike', exposure_type]].copy()
    calls_df = calls_df[calls_df[exposure_type] != 0]
    calls_df['OptionType'] = 'Call'

    puts_df = puts[['strike', exposure_type]].copy()
    puts_df = puts_df[puts_df[exposure_type] != 0]
    puts_df['OptionType'] = 'Put'

    # Calculate strike range around current price
    min_strike = S - st.session_state.strike_range
    max_strike = S + st.session_state.strike_range
    
    # Apply strike range filter
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    # Apply strike range filter
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Filter the original dataframes for net exposure calculation
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Calculate Net Exposure based on type using filtered data
    if exposure_type == 'GEX':
        if st.session_state.gex_type == 'Net':
            net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() - puts_filtered.groupby('strike')[exposure_type].sum()
        else:  # Absolute
            calls_gex = calls_filtered.groupby('strike')[exposure_type].sum()
            puts_gex = puts_filtered.groupby('strike')[exposure_type].sum()
            net_exposure = pd.Series(index=set(calls_gex.index) | set(puts_gex.index))
            for strike in net_exposure.index:
                call_val = abs(calls_gex.get(strike, 0))
                put_val = abs(puts_gex.get(strike, 0))
                net_exposure[strike] = call_val if call_val >= put_val else -put_val
    elif exposure_type == 'DEX':
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()
    else:  # VEX, Charm, Speed, Vomma
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()

    # Calculate total Greek values
    total_call_value = calls_df[exposure_type].sum()
    total_put_value = puts_df[exposure_type].sum()

    # Update title to include total Greek values with colored values using HTML
    title_with_totals = (
        f"{title}     "
        f"<span style='color: {st.session_state.call_color}'>{total_call_value:.0f}</span> | "
        f"<span style='color: {st.session_state.put_color}'>{total_put_value:.0f}</span>"
    )

    fig = go.Figure()

    # Add calls if enabled
    if (st.session_state.show_calls):
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_exposure.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_exposure.index,
                y=net_exposure.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_exposure.values]
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Update layout
    padding = st.session_state.strike_range * 0.1
    fig.update_layout(
        title=dict(
            text=title_with_totals,
            xref="paper",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)  # Title slightly larger
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text=title,
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='group',
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10),
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )

    fig = add_current_price_line(fig, S)
    return fig

def calculate_max_pain(calls, puts):
    """Calculate max pain points based on call and put options."""
    if calls.empty or puts.empty:
        return None, None, None, None, None

    unique_strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))
    total_pain_by_strike = {}
    call_pain_by_strike = {}
    put_pain_by_strike = {}

    for strike in unique_strikes:
        # Calculate call pain (loss to option writers)
        call_pain = calls[calls['strike'] <= strike]['openInterest'] * (strike - calls[calls['strike'] <= strike]['strike'])
        call_pain_sum = call_pain.sum()
        call_pain_by_strike[strike] = call_pain_sum
        
        # Calculate put pain (loss to option writers)
        put_pain = puts[puts['strike'] >= strike]['openInterest'] * (puts[puts['strike'] >= strike]['strike'] - strike)
        put_pain_sum = put_pain.sum()
        put_pain_by_strike[strike] = put_pain_sum
        
        total_pain_by_strike[strike] = call_pain_sum + put_pain_sum

    if not total_pain_by_strike:
        return None, None, None, None, None

    max_pain_strike = min(total_pain_by_strike.items(), key=lambda x: x[1])[0]
    call_max_pain_strike = min(call_pain_by_strike.items(), key=lambda x: x[1])[0]
    put_max_pain_strike = min(put_pain_by_strike.items(), key=lambda x: x[1])[0]
    
    return (max_pain_strike, call_max_pain_strike, put_max_pain_strike, 
            total_pain_by_strike, call_pain_by_strike, put_pain_by_strike)

def create_max_pain_chart(calls, puts, S):
    """Create a chart showing max pain analysis with separate call and put pain."""
    result = calculate_max_pain(calls, puts)
    if result is None:
        return None
    
    (max_pain_strike, call_max_pain_strike, put_max_pain_strike,
     total_pain_by_strike, call_pain_by_strike, put_pain_by_strike) = result

    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price
    min_strike = S - st.session_state.strike_range
    max_strike = S + st.session_state.strike_range
    padding = st.session_state.strike_range * 0.1
    
    fig = go.Figure()

    # Add total pain line
    if st.session_state.chart_type == 'Bar':
        fig.add_trace(go.Bar(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            name='Total Pain',
            marker_color='white'
        ))
    elif st.session_state.chart_type == 'Line':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='lines',
            name='Total Pain',
            line=dict(color='white', width=2)
        ))
    elif st.session_state.chart_type == 'Scatter':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='markers',
            name='Total Pain',
            marker=dict(color='white')
        ))
    else:  # Area
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            fill='tozeroy',
            name='Total Pain',
            line=dict(color='white', width=0.5),
            fillcolor='rgba(255, 255, 255, 0.3)'
        ))

    # Add call pain line
    if st.session_state.show_calls:
        fig.add_trace(go.Scatter(
            x=list(call_pain_by_strike.keys()),
            y=list(call_pain_by_strike.values()),
            name='Call Pain',
            line=dict(color=call_color, width=1, dash='dot')
        ))

    # Add put pain line
    if st.session_state.show_puts:
        fig.add_trace(go.Scatter(
            x=list(put_pain_by_strike.keys()),
            y=list(put_pain_by_strike.values()),
            name='Put Pain',
            line=dict(color=put_color, width=1, dash='dot')
        ))

    # Add vertical lines for different max pain points
    fig.add_vline(
        x=max_pain_strike,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"Total Max Pain: {max_pain_strike}",
        annotation_position="top"
    )

    if st.session_state.show_calls:
        fig.add_vline(
            x=call_max_pain_strike,
            line_dash="dash",
            line_color=call_color,
            opacity=0.7,
            annotation_text=f"Call Max Pain: {call_max_pain_strike}",
            annotation_position="top"
        )

    if st.session_state.show_puts:
        fig.add_vline(
            x=put_max_pain_strike,
            line_dash="dash",
            line_color=put_color,
            opacity=0.7,
            annotation_text=f"Put Max Pain: {put_max_pain_strike}",
            annotation_position="top"
        )

    # Add current price line
    fig.add_vline(
        x=S,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"{S}",
        annotation_position="bottom"
    )

    fig.update_layout(
        title=dict(
            text='Max Pain',
            font=dict(size=st.session_state.chart_text_size + 8)  # Title slightly larger
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Total Pain',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10),
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    # Add range slider consistent with other charts
    fig.update_xaxes(rangeslider=dict(visible=True))
    
    return fig

def get_nearest_expiry(available_dates):
    """Get the nearest expiration date from the list of available dates."""
    if not available_dates:
        return None
    
    today = datetime.now().date()
    future_dates = [datetime.strptime(date, '%Y-%m-%d').date() 
                   for date in available_dates 
                   if datetime.strptime(date, '%Y-%m-%d').date() >= today]
    
    if not future_dates:
        return None
    
    return min(future_dates).strftime('%Y-%m-%d')

def create_davi_chart(calls, puts, S):
    """Create Delta-Adjusted Value Index chart that matches other exposure charts style"""
    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate DAVI for calls and puts with filtering
    # Only keep non-zero values
    calls_df = calls.copy()
    puts_df = puts.copy()
    
    calls_df['DAVI'] = (calls_df['volume'] + calls_df['openInterest']) * calls_df['lastPrice'] * calls_df['calc_delta']
    calls_df = calls_df[calls_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    calls_df['OptionType'] = 'Call'

    puts_df['DAVI'] = (puts_df['volume'] + puts_df['openInterest']) * puts_df['lastPrice'] * puts_df['calc_delta']
    puts_df = puts_df[puts_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    puts_df['OptionType'] = 'Put'

    # Calculate strike range around current price
    min_strike = S - st.session_state.strike_range
    max_strike = S + st.session_state.strike_range
    
    # Filter data based on strike range
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Filter the original dataframes for net exposure calculation
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Calculate Net DAVI
    net_davi = pd.Series(0, index=sorted(set(calls_df['strike']) | set(puts_df['strike'])))
    if not calls_df.empty:
        net_davi = net_davi.add(calls_df.groupby('strike')['DAVI'].sum(), fill_value=0)
    if not puts_df.empty:
        net_davi = net_davi.add(puts_df.groupby('strike')['DAVI'].sum(), fill_value=0)

    # Calculate totals for title
    total_call_davi = calls_df['DAVI'].sum()
    total_put_davi = puts_df['DAVI'].sum()

    # Create title with totals
    title_with_totals = (
        f"Delta-Adjusted Value Index by Strike     "
        f"<span style='color: {call_color}'>{total_call_davi:.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_davi:.0f}</span>"
    )

    fig = go.Figure()

    # Add calls if enabled
    if st.session_state.show_calls:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_davi.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_davi.index,
                y=net_davi.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_davi.values]
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Add current price line
    fig = add_current_price_line(fig, S)

    # Update layout
    padding = st.session_state.strike_range * 0.1
    fig.update_layout(
        title=dict(
            text=title_with_totals,
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='DAVI',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='group',
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10),
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )

    return fig

if st.session_state.current_page == "OI & Volume":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        st.write("**Select filters below to see updated data, charts, and tables.**")
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="options_data_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_oi"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)  # Pass S to fetch_options_for_date
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                volume_over_oi = st.checkbox("Show only rows where Volume > Open Interest")
                # Filter data based on volume over OI if checked
                calls_filtered = all_calls.copy()
                puts_filtered = all_puts.copy()
                if volume_over_oi:
                    calls_filtered = calls_filtered[calls_filtered['volume'] > calls_filtered['openInterest']]
                    puts_filtered = puts_filtered[puts_filtered['volume'] > puts_filtered['openInterest']]
                if calls_filtered.empty and puts_filtered.empty:
                    st.warning("No data left after applying filters.")
                else:
                    charts_container = st.container()
                    tables_container = st.container()
                    with charts_container:
                        st.subheader(f"Options Data for {ticker} (Multiple Expiries)")
                        if not calls_filtered.empty and not puts_filtered.empty:
                            fig_oi, fig_volume = create_oi_volume_charts(calls_filtered, puts_filtered, S)
                            st.plotly_chart(fig_oi, use_container_width=True, key=f"Options Data_oi_chart")
                            st.plotly_chart(fig_volume, use_container_width=True, key=f"Options Data_volume_chart")
                        else:
                            st.warning("No data to chart for the chosen filters.")
                    with tables_container:
                        st.write("### Filtered Data Tables")
                        if not calls_filtered.empty:
                            st.write("**Calls Table**")
                            st.dataframe(calls_filtered)
                        else:
                            st.write("No calls match filters.")
                        if not puts_filtered.empty:
                            st.write("**Puts Table**")
                            st.dataframe(puts_filtered)
                        else:
                            st.write("No puts match filters.")

elif st.session_state.current_page == "Volume Ratio":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="volume_ratio_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_volume"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)  # Pass S to fetch_options_for_date
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                call_volume = all_calls['volume'].sum()
                put_volume = all_puts['volume'].sum()
                fig = create_donut_chart(call_volume, put_volume)
                st.plotly_chart(fig, use_container_width=True, key=f"Volume Ratio_donut_chart")
                st.markdown(f"**Total Call Volume:** {call_volume}")
                st.markdown(f"**Total Put Volume:** {put_volume}")

elif st.session_state.current_page == "Gamma Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Vanna Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Delta Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Charm Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Speed Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Vomma Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Calculated Greeks":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        st.write("This page calculates delta, gamma, and vanna based on market data.")
        
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="calculated_greeks_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_greeks"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                # Get nearest expiry and use it as default
                nearest_expiry = get_nearest_expiry(available_dates)
                expiry_date_str = st.selectbox(
                    "Select an Exp. Date:", 
                    options=available_dates, 
                    index=available_dates.index(nearest_expiry) if nearest_expiry else None, 
                    key="calculated_greeks_expiry_main"
                )
                
                if expiry_date_str:  # Only proceed if an expiry date is selected
                    
                    selected_expiry = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
                    calls, puts = fetch_options_for_date(ticker, expiry_date_str, S)
                    
                    if calls.empty and puts.empty:
                        st.warning("No options data available for this ticker.")
                        st.stop()

                    # Rest of the Calculated Greeks logic
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    
                    # Get stock price
                    stock = yf.Ticker(ticker)
                    S = get_last_price(stock)
                    if S is None:
                        st.error("Could not fetch underlying price.")
                        st.stop()

                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    # Change this condition to allow same-day expiration
                    if t_days < 0:  # Changed from t_days <= 0
                        st.error("The selected expiration date is in the past!")
                        st.stop()

                    t = t_days / 365.0
                    st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                    
                    def compute_row_greeks(row, flag):
                        try:
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})
                            
                            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
                            return pd.Series({
                                "calc_delta": delta_val,
                                "calc_gamma": gamma_val,
                                "calc_vanna": vanna_val
                            })
                        except Exception as e:
                            st.warning(f"Error calculating greeks: {str(e)}")
                            return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})

                    results = {}
                    
                    # Process calls
                    if not calls.empty:
                        try:
                            calls_copy = calls.copy()
                            greeks_calls = calls_copy.apply(lambda row: compute_row_greeks(row, "c"), axis=1)
                            results["Calls"] = pd.concat([calls_copy, greeks_calls], axis=1)
                        except Exception as e:
                            st.warning(f"Error processing calls: {str(e)}")
                    else:
                        st.warning("No call options data available.")

                    # Process puts
                    if not puts.empty:
                        try:
                            puts_copy = puts.copy()
                            greeks_puts = puts_copy.apply(lambda row: compute_row_greeks(row, "p"), axis=1)
                            results["Puts"] = pd.concat([puts_copy, greeks_puts], axis=1)
                        except Exception as e:
                            st.warning(f"Error processing puts: {str(e)}")
                    else:
                        st.warning("No put options data available.")

                    # Display results
                    for typ, df in results.items():
                        try:
                            st.write(f"### {typ} with Calculated Greeks")
                            st.dataframe(df[['contractSymbol', 'strike', 'impliedVolatility', 'calc_delta', 'calc_gamma', 'calc_vanna']])
                            fig = px.scatter(df, x="strike", y="calc_delta", title=f"{typ}: Delta vs. Strike",
                                         labels={"strike": "Strike", "calc_delta": "Calculated Delta"})
                            st.plotly_chart(fig, use_container_width=True, key=f"Calculated Greeks_{typ.lower()}_scatter")
                        except Exception as e:
                            st.error(f"Error displaying {typ} data: {str(e)}")
                else:
                    st.warning("Please select an expiration date to view the calculations.")
                    st.stop()

elif st.session_state.current_page == "Dashboard":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="dashboard_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_dashboard"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                # Get nearest expiry and use it as default
                nearest_expiry = get_nearest_expiry(available_dates)
                expiry_date_str = st.selectbox(
                    "Select an Exp. Date:", 
                    options=available_dates,
                    index=available_dates.index(nearest_expiry) if nearest_expiry else None,
                    key="dashboard_expiry_main"
                )
                
                if expiry_date_str:  # Only proceed if expiry date is selected
                    calls, puts, _, t, selected_expiry, today = compute_greeks_and_charts(ticker, expiry_date_str, "dashboard", S)
                    if calls is None or puts is None:
                        st.stop()
                        

                    fig_gamma = create_exposure_bar_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S)
                    fig_vanna = create_exposure_bar_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S)
                    fig_delta = create_exposure_bar_chart(calls, puts, "DEX", "Delta Exposure by Strike", S)
                    fig_charm = create_exposure_bar_chart(calls, puts, "Charm", "Charm Exposure by Strike", S)
                    fig_speed = create_exposure_bar_chart(calls, puts, "Speed", "Speed Exposure by Strike", S)
                    fig_vomma = create_exposure_bar_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S)
                    
                    # Intraday price chart
                    intraday_data, current_price, vix_data = get_combined_intraday_data(ticker)
                    if intraday_data is None or current_price is None:
                        st.warning("No intraday data available for this ticker.")
                    else:
                        # Initialize plot with cleared shapes/annotations
                        fig_intraday = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_intraday.layout.shapes = []
                        fig_intraday.layout.annotations = []
                        
                        # Add either candlestick or line trace based on selection
                        if st.session_state.intraday_chart_type == 'Candlestick':
                            # Set colors based on candlestick type
                            if st.session_state.candlestick_type == 'Hollow':
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing=dict(
                                            line=dict(color=st.session_state.call_color),
                                            fillcolor='rgba(0,0,0,0)'
                                        ),
                                        decreasing=dict(
                                            line=dict(color=st.session_state.put_color),
                                            fillcolor='rgba(0,0,0,0)'
                                        )
                                    ),
                                    secondary_y=False
                                )
                            else:  # Filled candlesticks
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing_line_color=st.session_state.call_color,
                                        decreasing_line_color=st.session_state.put_color,
                                        increasing_fillcolor=st.session_state.call_color,
                                        decreasing_fillcolor=st.session_state.put_color
                                    ),
                                    secondary_y=False
                                )
                        else:  # Line chart remains the same
                            fig_intraday.add_trace(
                                go.Scatter(
                                    x=intraday_data.index,
                                    y=intraday_data['Close'],
                                    name="Price",
                                    line=dict(color='gold')
                                ),
                                secondary_y=False
                            )
                        
                        # Add VIX overlay if enabled
                        if st.session_state.show_vix_overlay and vix_data is not None and not vix_data.empty:
                            # Normalize VIX data to match price scale
                            price_min = intraday_data['Low'].min()
                            price_max = intraday_data['High'].max()
                            price_range = price_max - price_min
                            
                            vix_min = vix_data['Close'].min()
                            vix_max = vix_data['Close'].max()
                            vix_range = vix_max - vix_min
                            
                            # Scale VIX values to match price range
                            normalized_vix = price_min + ((vix_data['Close'] - vix_min) * price_range / vix_range)
                            
                            fig_intraday.add_trace(
                                go.Scatter(
                                    x=vix_data.index,
                                    y=normalized_vix,
                                    name='VIX',
                                    line=dict(color=st.session_state.vix_color),
                                    opacity=0.7
                                ),
                                secondary_y=False  # Changed to False to use same y-axis
                            )
                            
                            # Add VIX value annotations on the right side
                            fig_intraday.add_annotation(
                                x=vix_data.index[-1],
                                y=normalized_vix.iloc[-1],
                                text=f"{vix_data['Close'].iloc[-1]:.2f}",
                                showarrow=False,
                                arrowhead=1,
                                xshift=16,
                                ax=50,
                                ay=0,
                                font=dict(color=st.session_state.vix_color, size=15)
                            )

                        # Only add price annotation if we have a valid price
                        if current_price is not None:
                            fig_intraday.add_annotation(
                                x=intraday_data.index[-1],
                                y=current_price,
                                xref='x',
                                yref='y',
                                xshift=27,
                                showarrow=False,
                                text=f"{current_price:,.2f}",
                                font=dict(color='yellow', size=15)
                            )
                            
                            # Calculate y-axis range with less aggressive padding
                            market_open_price = intraday_data['Open'].iloc[0]
                            price_range = max(intraday_data['High'].max(), current_price) - min(intraday_data['Low'].min(), current_price)
                            padding = price_range * 0.05  # Use 5% padding instead
                            
                            y_min = min(intraday_data['Low'].min(), current_price, market_open_price) - padding
                            y_max = max(intraday_data['High'].max(), current_price, market_open_price) + padding
                            
                            # Ensure there's always some minimum range to prevent crushing
                            if abs(y_max - y_min) < (current_price * 0.001):  # Minimum 0.1% range
                                center = (y_max + y_min) / 2
                                y_min = center * 0.999
                                y_max = center * 1.001
                            
                            fig_intraday.update_yaxes(range=[y_min, y_max])

                        # Process options data
                        calls['OptionType'] = 'Call'
                        puts['OptionType'] = 'Put'
                        
                        # Combine and filter GEX data
                        options_df = pd.concat([calls, puts]).dropna(subset=['GEX'])
                        added_strikes = set()
                        
                        if options_df.empty:
                            st.warning("Intraday Data will display near market open.")
                        else:
                            # Get all GEX levels sorted by absolute value and distance from current price
                            top5 = options_df.nlargest(5, 'GEX')[['strike', 'GEX', 'OptionType']]
                            
                            # Sort by distance from current price for zoom calculation
                            top5['distance'] = abs(top5['strike'] - current_price)
                            nearest_3 = top5.nsmallest(3, 'distance')
                            
                            # Find max GEX value for color scaling using all top 5
                            max_gex = abs(top5['GEX']).max()
                            
                            # Add all GEX levels
                            for row in top5.itertuples():
                                if row.strike not in added_strikes:
                                    # Ensure intensity is not NaN and within valid range
                                    if not pd.isna(row.GEX) and row.GEX != 0:
                                        intensity = 0.4 + (min(abs(row.GEX) / max_gex, 1.0) * 0.6)
                                        
                                        if not pd.isna(intensity) and 0 <= intensity <= 1:
                                            if row.OptionType == 'Call':
                                                base_color = st.session_state.call_color
                                            else:
                                                base_color = st.session_state.put_color
                                                
                                            rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                                            color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})'
                                            
                                            fig_intraday.add_shape(
                                                type='line',
                                                x0=intraday_data.index[0],
                                                x1=intraday_data.index[-1],
                                                y0=row.strike,
                                                y1=row.strike,
                                                line=dict(color=color, width=2),
                                                xref='x',
                                                yref='y',
                                                layer='below'
                                            )
                                            
                                            fig_intraday.add_annotation(
                                                x=intraday_data.index[-1],
                                                y=row.strike,
                                                xref='x',
                                                yref='y',
                                                showarrow=True,
                                                arrowhead=1,
                                                text=f"GEX {row.GEX:,.0f}",
                                                font=dict(color=color),
                                                arrowcolor=color
                                            )
                                            
                                            added_strikes.add(row.strike)
                            
                            # Set y-axis range based on price data and nearest 3 GEX levels
                            y_min = min(
                                min(intraday_data['Low'].min(), current_price),
                                nearest_3['strike'].min()
                            ) * 0.9995
                            
                            y_max = max(
                                max(intraday_data['High'].max(), current_price),
                                nearest_3['strike'].max()
                            ) * 1.0005

                    
                            # Update layout with interactive features
                            fig_intraday.update_layout(
                                title=f"Intraday Price for {ticker}",
                                height=600,
                                hovermode='x unified',
                                margin=dict(r=150, l=50),
                                xaxis=dict(
                                    autorange=True,
                                    rangeslider=dict(visible=False)
                                ),
                                yaxis=dict(
                                    range=[y_min, y_max],
                                    autorange=False,
                                    fixedrange=False
                                )
                            )

                    
                    # Calculate volume ratio
                    call_volume = calls['volume'].sum()
                    put_volume = puts['volume'].sum()
                    fig_volume_ratio = create_donut_chart(call_volume, put_volume)
                    
                    
                    fig_max_pain = create_max_pain_chart(calls, puts, S)
                    
                    
                    chart_options = [
                        "Intraday Price", 
                        "Gamma Exposure", 
                        "Vanna Exposure", 
                        "Delta Exposure", 
                        "Charm Exposure", 
                        "Speed Exposure", 
                        "Vomma Exposure", 
                        "Volume Ratio",
                        "Max Pain",
                        "Delta-Adjusted Value Index" 
                    ]
                    default_charts = [
                         "Intraday Price",
                         "Gamma Exposure",
                         "Vanna Exposure",
                         "Delta Exposure",
                         "Charm Exposure"
                     ]
                    selected_charts = st.multiselect("Select charts to display:", chart_options, default=[
                         chart for chart in default_charts if chart in chart_options
                     ])

                    
                    if 'saved_ticker' in st.session_state and st.session_state.saved_ticker:
                        current_price = get_current_price(st.session_state.saved_ticker)
                        if current_price:
                            gainers_df = get_screener_data("day_gainers")
                            losers_df = get_screener_data("day_losers")
                            
                            if not gainers_df.empty and not losers_df.empty:
                                market_text = (
                                    "<span style='color: gray; font-size: 14px;'>Gainers:</span> "
                                    + " ".join([
                                        f"<span style='color: {st.session_state.call_color}'>"
                                        f"{gainer['symbol']}: +{gainer['regularMarketChangePercent']:.1f}%</span> "
                                        for _, gainer in gainers_df.head().iterrows()
                                    ])
                                    + " | <span style='color: gray; font-size: 14px;'>Losers:</span> "
                                    + " ".join([
                                        f"<span style='color: {st.session_state.put_color}'>"
                                        f"{loser['symbol']}: {loser['regularMarketChangePercent']:.1f}%</span> "
                                        for _, loser in losers_df.head().iterrows()
                                    ])
                                )
                                st.markdown(market_text, unsafe_allow_html=True)
                            
                            st.markdown(f"#### Current Price: ${current_price:.2f}")
                            st.markdown("---")

                    
                    # Display selected charts
                    if "Intraday Price" in selected_charts:
                        st.plotly_chart(fig_intraday, use_container_width=True, key="Dashboard_intraday_chart")
                    
                    # Create a list of selected supplemental charts
                    supplemental_charts = []
                    if "Gamma Exposure" in selected_charts:
                        supplemental_charts.append(fig_gamma)
                    if "Delta Exposure" in selected_charts:
                        supplemental_charts.append(fig_delta)
                    if "Vanna Exposure" in selected_charts:
                        supplemental_charts.append(fig_vanna)
                    if "Charm Exposure" in selected_charts:
                        supplemental_charts.append(fig_charm)
                    if "Speed Exposure" in selected_charts:
                        supplemental_charts.append(fig_speed)
                    if "Vomma Exposure" in selected_charts:
                        supplemental_charts.append(fig_vomma)
                    if "Volume Ratio" in selected_charts:
                        supplemental_charts.append(fig_volume_ratio)
                    if "Max Pain" in selected_charts:
                        supplemental_charts.append(fig_max_pain)
                    if "Delta-Adjusted Value Index" in selected_charts:
                        supplemental_charts.append(create_davi_chart(calls, puts, S))

                    
                    # Display supplemental charts in rows of 2
                    for i in range(0, len(supplemental_charts), 2):
                        cols = st.columns(2)
                        for j, chart in enumerate(supplemental_charts[i:i+2]):
                            if chart is not None: 
                                cols[j].plotly_chart(chart, use_container_width=True)

                else:
                    st.warning("Please select an expiration date to view the dashboard.")
                    st.stop()

elif st.session_state.current_page == "Max Pain":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="max_pain_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_max_pain"):
                st.cache_data.clear()
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)  # Pass S to fetch_options_for_date
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                # Calculate and display max pain
                result = calculate_max_pain(all_calls, all_puts)
                if result is not None:
                    max_pain_strike, call_max_pain_strike, put_max_pain_strike, *_ = result
                    st.markdown(f"### Total Max Pain Strike: ${max_pain_strike:.2f}")
                    st.markdown(f"### Call Max Pain Strike: ${call_max_pain_strike:.2f}")
                    st.markdown(f"### Put Max Pain Strike: ${put_max_pain_strike:.2f}")
                    st.markdown(f"### Current Price: ${S:.2f}")
                    st.markdown(f"### Distance to Max Pain: ${abs(S - max_pain_strike):.2f}")
                    
                    # Create and display the max pain chart
                    fig = create_max_pain_chart(all_calls, all_puts, S)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not calculate max pain point.")

if st.session_state.get('current_page') == "IV Surface":
    main_container = st.container()
    with main_container:
        # Layout for ticker input and refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input(
                "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
                value=st.session_state.get('saved_ticker', ''),
                key="iv_skew_ticker"
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄", key="refresh_button_skew"):
                st.cache_data.clear()
                st.rerun()

        # Format and save ticker
        ticker = format_ticker(user_ticker)
        if ticker != st.session_state.get('saved_ticker', ''):
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = yf.Ticker(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Multiselect for expiration dates with no default selection
            selected_expiry_dates = st.multiselect(
                "Select Expiration Dates (1 for 2D chart, 2+ for 3D surface):",
                options=available_dates,
                default=None,  # Explicitly set to None to avoid pre-selection
                key="iv_date_selector"
            )

            # Store the user's selection
            st.session_state.iv_selected_dates = selected_expiry_dates

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all IV data

                    # Calculate strike range using strike_range setting
                    strike_range = st.session_state.strike_range
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    for exp_date in selected_expiry_dates:
                        expiry_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        days_to_exp = (expiry_date - datetime.now().date()).days
                        calls, puts = fetch_options_for_date(ticker, exp_date, S)

                        # Filter strikes within range
                        calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                        puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

                        for strike in sorted(set(calls['strike'].unique()) | set(puts['strike'].unique())):
                            call_iv = calls[calls['strike'] == strike]['impliedVolatility'].mean()
                            put_iv = puts[puts['strike'] == strike]['impliedVolatility'].mean()
                            iv = np.nanmean([call_iv, put_iv])
                            if not np.isnan(iv):
                                all_data.append({
                                    'strike': strike,
                                    'days': days_to_exp,
                                    'iv': iv * 100  # Convert to percentage
                                })

                    if not all_data:
                        st.warning("No valid IV data available within strike range.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    # Create custom colorscale using call/put colors
                    call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    custom_colorscale = [
                        [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                        [0.5, 'rgb(255, 215, 0)'],  # Gold at center
                        [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                    ]

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot
                        fig = go.Figure()

                        # Filter data for single expiration
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Calculate center IV for coloring
                        center_iv = single_date_df['iv'].median()

                        # Create line segments with color gradient based on IV value
                        for i in range(len(single_date_df) - 1):
                            iv_val = single_date_df['iv'].iloc[i]
                            if iv_val >= center_iv:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                            
                            fig.add_trace(go.Scatter(
                                x=single_date_df['strike'].iloc[i:i+2],
                                y=single_date_df['iv'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
                            ))

                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )

                        # Update layout
                        padding = strike_range * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Implied Volatility (%)',
                            yaxis=dict(tickformat='.1f', ticksuffix='%'),
                            xaxis=dict(range=[min_strike - padding, max_strike + padding]),
                            width=800,
                            height=600
                        )

                    else:
                        # 3D Surface Plot
                        # Create meshgrid for interpolation
                        unique_strikes = np.linspace(min_strike, max_strike, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Interpolate surface
                        Z = griddata(
                            (df['strike'], df['days']),
                            df['iv'],
                            (X, Y),
                            method='linear',
                            fill_value=np.nan
                        )

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add IV surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=custom_colorscale,
                            colorbar=dict(
                                title=dict(text='IV %', side='right'),
                                tickformat='.1f',
                                ticksuffix='%'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['iv'].min(), df['iv'].max()], [df['iv'].min(), df['iv'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Implied Volatility (%)', tickformat='.1f', ticksuffix='%')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.get('current_page') == "GEX Surface":
    main_container = st.container()
    with main_container:
        # Layout for ticker input and refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input(
                "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
                value=st.session_state.get('saved_ticker', ''),
                key="gex_surface_ticker"
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄", key="refresh_button_gex"):
                st.cache_data.clear()
                st.rerun()

        # Format and save ticker
        ticker = format_ticker(user_ticker)
        if ticker != st.session_state.get('saved_ticker', ''):
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = yf.Ticker(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Multiselect for expiration dates with no default selection
            selected_expiry_dates = st.multiselect(
                "Select Expiration Dates (1 for 2D chart, 2+ for 3D surface):",
                options=available_dates,
                default=None,
                key="gex_date_selector"
            )

            # Store the user's selection
            st.session_state.gex_selected_dates = selected_expiry_dates

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all computed GEX data

                    # Calculate strike range using strike_range setting
                    strike_range = st.session_state.strike_range
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    for date in selected_expiry_dates:
                        # Compute greeks using the same function as gamma exposure chart
                        calls, puts, _, t, selected_expiry, today = compute_greeks_and_charts(ticker, date, "gex", S)
                        
                        if calls is not None and puts is not None:
                            days_to_exp = (selected_expiry - today).days
                            
                            # Filter and process data within strike range
                            calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                            puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
                            
                            for _, row in calls.iterrows():
                                if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                    all_data.append({
                                        'strike': row['strike'],
                                        'days': days_to_exp,
                                        'gex': row['GEX']
                                    })
                            
                            for _, row in puts.iterrows():
                                if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                    all_data.append({
                                        'strike': row['strike'],
                                        'days': days_to_exp,
                                        'gex': -row['GEX']
                                    })

                    if not all_data:
                        st.warning("No valid GEX data available.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot for single expiration
                        fig = go.Figure()
                        
                        # Filter data for the single expiration date
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Group by strike and sum GEX values
                        grouped_gex = single_date_df.groupby('strike')['gex'].sum().reset_index()
                        
                        # Create line plot with color gradient based on GEX sign
                        for i in range(len(grouped_gex) - 1):
                            if grouped_gex['gex'].iloc[i] >= 0:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                                
                            fig.add_trace(go.Scatter(
                                x=grouped_gex['strike'].iloc[i:i+2],
                                y=grouped_gex['gex'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>GEX: %{y:,.0f}<extra></extra>'
                            ))
                        
                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )
                        
                        # Update layout with adjusted range
                        padding = (max_strike - min_strike) * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Profile - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Gamma Exposure',
                            width=800,
                            height=600,
                            xaxis=dict(range=[min_strike - padding, max_strike + padding])
                        )

                    else:
                        # 3D Surface Plot for multiple expirations
                        # Create meshgrid with adjusted strike range
                        padding = (max_strike - min_strike) * 0.05
                        unique_strikes = np.linspace(min_strike - padding, max_strike + padding, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Aggregate GEX values by strike and days
                        df_grouped = df.groupby(['strike', 'days'])['gex'].sum().reset_index()

                        # Interpolation
                        Z = griddata(
                            (df_grouped['strike'], df_grouped['days']),
                            df_grouped['gex'],
                            (X, Y),
                            method='linear',
                            fill_value=0
                        )

                        # Create custom colorscale using call/put colors
                        call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        
                        colorscale = [
                            [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                            [0.5, 'rgb(255, 215, 0)'],  # Gold at zero
                            [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                        ]

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add GEX surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=colorscale,
                            opacity=1.0,
                            colorbar=dict(
                                title=dict(text='Net GEX', side='right'),
                                tickformat=',.0f'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>Net GEX: %{z:,.0f}<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['gex'].min(), df['gex'].max()], [df['gex'].min(), df['gex'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Gamma Exposure', tickformat=',.0f')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.current_page == "Analysis":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="analysis_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_analysis"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            # Fetch 6 months of historical data instead of 1 year
            historical_data = stock.history(period="6mo", interval="1d")  # Changed from "1y" to "6mo"
            
            if historical_data.empty:
                st.warning("No historical data available for this ticker.")
                st.stop()

            # Calculate indicators with proper padding
            lookback = 20  # Standard lookback period
            padding_data = pd.concat([
                historical_data['Close'].iloc[:lookback].iloc[::-1],  # Reverse first lookback periods
                historical_data['Close']
            ])
            
            # Calculate SMA and Bollinger Bands with padding
            sma_padded = padding_data.rolling(window=lookback).mean()
            std_padded = padding_data.rolling(window=lookback).std()
            
            # Trim padding and assign to historical_data
            historical_data['SMA'] = sma_padded[lookback:].values
            historical_data['Upper Band'] = historical_data['SMA'] + 2 * std_padded[lookback:].values
            historical_data['Lower Band'] = historical_data['SMA'] - 2 * std_padded[lookback:].values

            def calculate_rsi(data, period=14):
                # Add padding for RSI calculation
                padding = pd.concat([data.iloc[:period].iloc[::-1], data])
                delta = padding.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                # Return only the non-padded portion
                return rsi[period:].values

            historical_data['RSI'] = calculate_rsi(historical_data['Close'])
            
            # Calculate GEX with padding for rolling average
            historical_data['GEX'] = historical_data['Close'].pct_change() * 1000
            gex_padded = pd.concat([
                historical_data['GEX'].iloc[:lookback].iloc[::-1],
                historical_data['GEX']
            ])
            historical_data['Rolling GEX Avg'] = gex_padded.rolling(window=lookback).mean()[lookback:].values

            # Create subplots with independent X-axes
            fig = make_subplots(
                rows=3, 
                cols=1, 
                shared_xaxes=False,  # Changed to False for independent X-axes
                vertical_spacing=0.1,
                subplot_titles=(
                    'Price vs. Simple Moving Average and Bollinger Bands',
                    'GEX Rolling Average',
                    'RSI'
                )
            )

            # Get colors from session state
            call_color = st.session_state.call_color
            put_color = st.session_state.put_color

            # Price and indicators with consistent colors
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index, 
                    y=historical_data['Close'], 
                    name='Price', 
                    line=dict(color='gold')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index, 
                    y=historical_data['SMA'], 
                    name='SMA', 
                    line=dict(color='purple')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index, 
                    y=historical_data['Upper Band'], 
                    name='Upper Band',
                    line=dict(color=call_color, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index, 
                    y=historical_data['Lower Band'], 
                    name='Lower Band',
                    line=dict(color=put_color, dash='dash')
                ),
                row=1, col=1
            )

            # GEX with consistent colors
            positive_gex = historical_data['GEX'] >= 0
            negative_gex = historical_data['GEX'] < 0

            # Add positive GEX bars
            fig.add_trace(
                go.Bar(
                    x=historical_data.index[positive_gex],
                    y=historical_data['GEX'][positive_gex],
                    name='Positive GEX',
                    marker_color=call_color
                ),
                row=2, col=1
            )

            # Add negative GEX bars
            fig.add_trace(
                go.Bar(
                    x=historical_data.index[negative_gex],
                    y=historical_data['GEX'][negative_gex],
                    name='Negative GEX',
                    marker_color=put_color
                ),
                row=2, col=1
            )

            # Add rolling average
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Rolling GEX Avg'],
                    name='Rolling Average',
                    line=dict(color='magenta')
                ),
                row=2, col=1
            )

            # RSI with consistent colors
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['RSI'],
                    name='RSI',
                    line=dict(color='turquoise')
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color=call_color,
                row=3,
                col=1,
                annotation_text="Overbought"
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color=put_color,
                row=3,
                col=1,
                annotation_text="Oversold"
            )

            # Update layout with consistent styling
            fig.update_layout(
                template="plotly_dark",
                title=dict(
                    text=f"Technical Analysis for {ticker}",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 8)
                ),
                showlegend=True,
                height=1000,
                legend=dict(
                    font=dict(size=st.session_state.chart_text_size)
                )
            )

            # Update X-axes for each subplot to show dates
            for row in [1, 2, 3]:
                fig.update_xaxes(
                    showticklabels=True,  # Ensure dates are visible
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_text="Date",  # Optional: Add "Date" label to each X-axis
                    title_font=dict(size=st.session_state.chart_text_size),
                    row=row,
                    col=1
                )

            # Update Y-axes with consistent styling
            fig.update_yaxes(
                title_text="Price",
                row=1,
                col=1,
                tickfont=dict(size=st.session_state.chart_text_size),
                title_font=dict(size=st.session_state.chart_text_size)
            )
            fig.update_yaxes(
                title_text="GEX",
                row=2,
                col=1,
                tickfont=dict(size=st.session_state.chart_text_size),
                title_font=dict(size=st.session_state.chart_text_size)
            )
            fig.update_yaxes(
                title_text="RSI",
                range=[0, 100],
                row=3,
                col=1,
                tickfont=dict(size=st.session_state.chart_text_size),
                title_font=dict(size=st.session_state.chart_text_size)
            )

            st.plotly_chart(fig, use_container_width=True)
    st.stop()

elif st.session_state.current_page == "Delta-Adjusted Value Index":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="davi_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_davi"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, "davi", S)[:2]
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                fig = create_davi_chart(all_calls, all_puts, S)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# Auto-refresh
# -----------------------------------------
refresh_rate = float(st.session_state.get('refresh_rate', 10))  # Convert to float
if not st.session_state.get("loading_complete", False):
    st.session_state.loading_complete = True
    st.rerun()
else:
    time.sleep(refresh_rate)
    st.rerun()