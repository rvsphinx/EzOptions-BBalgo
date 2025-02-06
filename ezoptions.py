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

st.set_page_config(layout="wide")

# -------------------------------
# Helper Functions
# -------------------------------
def format_ticker(ticker):
    """Helper function to format tickers for indices"""
    ticker = ticker.upper()
    if ticker == "SPX":
        return "%5ESPX"
    elif ticker == "NDX":
        return "%5ENDX"
    elif ticker == "VIX":
        return "^VIX"
    return ticker

@st.cache_data(ttl=10)
def fetch_options_for_date(ticker, date):
    """
    Fetches option chains for the given ticker and date.
    Returns two DataFrames: one for calls and one for puts.
    """
    print(f"Fetching option chain for {ticker} EXP {date}")
    stock = yf.Ticker(ticker)
    try:
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

# -------------------------------
# Fetch all options experations and add extract expiry
# -------------------------------
@st.cache_data(ttl=10)
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
        # Fallback for tickers like SPX which return an empty options list
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
@st.cache_data(ttl=10)
def get_current_price(ticker):
    """
    Fetches the current price for the given ticker.
    """
    print(f"Fetching current price for {ticker}")
    stock = yf.Ticker(ticker)
    S = stock.info.get("regularMarketPrice")
    if S is None:
        S = stock.fast_info.get("lastPrice")
    return S

def create_oi_volume_charts(calls, puts):
    # Get underlying price
    S = get_current_price(ticker)
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
    
    fig_oi.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10)
        )
    )
    fig_oi.update_xaxes(rangeslider=dict(visible=True))
    
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
    
    fig_volume.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10)
        )
    )
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
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        showlegend=True
    )
    fig.update_traces(hoverinfo='label+percent+value', marker=dict(colors=[call_color, put_color]))  # Use custom colors
    return fig

# Greek Calculations
def calculate_greeks(flag, S, K, t, sigma):
    """
    Calculate delta, gamma and vanna for an option.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # These are correct from py_vollib
        delta_val = bs_delta(flag, S, K, t, 0, sigma)
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)
        vega_val = bs_vega(flag, S, K, t, 0, sigma)
        
        # Correct vanna calculation
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
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        norm_d1 = norm.pdf(d1)
        
        # Correct charm formula
        if flag == 'c':
            charm = -norm_d1 * (2*(0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
        else:  # put
            charm = -norm_d1 * (2*(0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
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
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        
        # Correct speed formula
        gamma = bs_gamma(flag, S, K, t, 0, sigma)
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
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Correct vomma formula
        vega = bs_vega(flag, S, K, t, 0, sigma)
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
        'saved_expiry_date',
        'call_color',  # Add color settings to preserved keys
        'put_color',
        'show_calls',  # Preserve visibility settings
        'show_puts',
        'show_net',
        'strike_range'  # Preserve strike range setting
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

def handle_page_change(new_page):
    """Handle page navigation and state management"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = new_page
        return True
    
    if st.session_state.current_page != new_page:
        st.session_state.current_page = new_page
        reset_session_state()
        return True
    
    return False

# Save selected ticker
def save_ticker(ticker):
    st.session_state.saved_ticker = ticker

# Save experation
def save_expiry_date(expiry_date):
    st.session_state.saved_expiry_date = expiry_date

st.sidebar.title("Navigation")
pages = ["Dashboard", "Volume Ratio", "OI & Volume", "Gamma Exposure", "Delta Exposure", 
          "Vanna Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Calculated Greeks"]

new_page = st.sidebar.radio("Select a page:", pages)

if handle_page_change(new_page):
     st.rerun()

# Add after st.sidebar.title("Navigation")
with st.sidebar.expander("Chart Settings", expanded=False):
    # Initialize session state colors if not already set
    if 'call_color' not in st.session_state:
        st.session_state.call_color = '#00FF00'  # Default green for calls
    if 'put_color' not in st.session_state:
        st.session_state.put_color = '#FF0000'  # Default red for puts
    
    # Initialize visibility settings if not already set
    if 'show_calls' not in st.session_state:
        st.session_state.show_calls = True
    if 'show_puts' not in st.session_state:
        st.session_state.show_puts = True
    if 'show_net' not in st.session_state:
        st.session_state.show_net = True

    st.write("Colors:")
    # Color pickers
    call_color = st.color_picker("Calls", st.session_state.call_color)
    put_color = st.color_picker("Puts", st.session_state.put_color)

    # Update session state when colors change
    if call_color != st.session_state.call_color:
        st.session_state.call_color = call_color
    if put_color != st.session_state.put_color:
        st.session_state.put_color = put_color

    st.write("Show/Hide Elements:")
    # Visibility toggles
    show_calls = st.checkbox("Show Calls", value=st.session_state.show_calls)
    show_puts = st.checkbox("Show Puts", value=st.session_state.show_puts)
    show_net = st.checkbox("Show Net", value=st.session_state.show_net)

    # Update session state when visibility changes
    st.session_state.show_calls = show_calls
    st.session_state.show_puts = show_puts
    st.session_state.show_net = show_net

    st.write("Strike Range:")
    # Initialize strike range in session state
    if 'strike_range' not in st.session_state:
        st.session_state.strike_range = 20.0  # Default range of ±20 strikes
    
    # Add strike range control
    strike_range = st.number_input(
        "Strike Range (±)",
        min_value=1.0,
        max_value=200.0,
        value=st.session_state.strike_range,
        step=1.0,
        key="strike_range_sidebar"
    )
    st.session_state.strike_range = strike_range

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

def compute_greeks_and_charts(ticker, expiry_date_str, page_key):
    calls, puts = fetch_options_for_date(ticker, expiry_date_str)
    if calls.empty and puts.empty:
        st.warning("No options data available for this ticker.")
        return None, None, None, None, None, None

    combined = pd.concat([calls, puts])
    combined = combined.dropna(subset=['extracted_expiry'])
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    calls = calls[calls['extracted_expiry'] == selected_expiry]
    puts = puts[puts['extracted_expiry'] == selected_expiry]

    S = get_current_price(ticker)
    if S is None:
        st.error("Could not fetch underlying price.")
        return None, None, None, None, None, None

    S = round(S, 2)
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
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Filter the original dataframes for net exposure calculation
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Calculate Net Exposure based on type using filtered data
    if exposure_type == 'DEX':
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()
    elif exposure_type == 'GEX':
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() - puts_filtered.groupby('strike')[exposure_type].sum()
    else:
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()

    fig = go.Figure()

    # Add calls if enabled
    if st.session_state.show_calls:
        fig.add_trace(go.Bar(
            x=calls_df['strike'],
            y=calls_df[exposure_type],
            name='Call',
            marker_color=call_color
        ))

    # Add puts if enabled
    if st.session_state.show_puts:
        fig.add_trace(go.Bar(
            x=puts_df['strike'],
            y=puts_df[exposure_type],
            name='Put',
            marker_color=put_color
        ))

    # Add Net if enabled
    if st.session_state.show_net and not net_exposure.empty:
        net_colors = [call_color if val >= 0 else put_color for val in net_exposure.values]
        fig.add_trace(go.Bar(
            x=net_exposure.index,
            y=net_exposure.values,
            name='Net',
            marker=dict(color=net_colors)
        ))

    # Update layout to center around current price
    padding = st.session_state.strike_range * 0.1  # Add 10% padding
    fig.update_layout(
        title=title,
        xaxis_title='Strike Price',
        yaxis_title=title,
        barmode='group',
        hovermode='x unified',
        xaxis=dict(
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil(st.session_state.strike_range / 10)  # Dynamic tick spacing
        )
    )

    fig = add_current_price_line(fig, S)
    return fig

if st.session_state.current_page == "OI & Volume":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        st.write("**Select filters below to see updated data, charts, and tables.**")
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="options_data_ticker")
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, index=available_dates.index(saved_expiry_date) if saved_expiry_date in available_dates else 0)
                save_expiry_date(expiry_date_str)  # Save the expiry date
                calls, puts = fetch_options_for_date(ticker, expiry_date_str)
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                else:
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    if calls.empty and puts.empty:
                        st.warning("No options data found for the selected expiry.")
                    else:
                        volume_over_oi = st.checkbox("Show only rows where Volume > Open Interest")
                        # Filter data based on volume over OI if checked
                        calls_filtered = calls.copy()
                        puts_filtered = puts.copy()
                        if volume_over_oi:
                            calls_filtered = calls_filtered[calls_filtered['volume'] > calls_filtered['openInterest']]
                            puts_filtered = puts_filtered[puts_filtered['volume'] > puts_filtered['openInterest']]
                        if calls_filtered.empty and puts_filtered.empty:
                            st.warning("No data left after applying filters.")
                        else:
                            charts_container = st.container()
                            tables_container = st.container()
                            with charts_container:
                                st.subheader(f"Options Data for {ticker} (Expiry: {expiry_date_str})")
                                if not calls_filtered.empty and not puts_filtered.empty:
                                    fig_oi, fig_volume = create_oi_volume_charts(calls_filtered, puts_filtered)
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
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="volume_ratio_ticker")
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, index=available_dates.index(saved_expiry_date) if saved_expiry_date in available_dates else 0, key="volume_ratio_expiry_main")
                save_expiry_date(expiry_date_str)  # Save the expiry date
                calls, puts = fetch_options_for_date(ticker, expiry_date_str)
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                else:
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    if calls.empty and puts.empty:
                        st.warning("No options data found for the selected expiry.")
                    else:
                        call_volume = calls['volume'].sum()
                        put_volume = puts['volume'].sum()
                        fig = create_donut_chart(call_volume, put_volume)
                        st.plotly_chart(fig, use_container_width=True, key=f"Volume Ratio_donut_chart")
                        st.markdown(f"**Total Call Volume:** {call_volume}")
                        st.markdown(f"**Total Put Volume:** {put_volume}")

elif st.session_state.current_page in ["Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure"]:
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        user_ticker = st.text_input(
            "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", 
            saved_ticker, 
            key=f"{page_name}_exposure_ticker"
        )
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, index=available_dates.index(saved_expiry_date) if saved_expiry_date in available_dates else 0, key=f"{page_name}_expiry_main")
                save_expiry_date(expiry_date_str)  # Save the expiry date
                calls, puts, S, t, selected_expiry, today = compute_greeks_and_charts(ticker, expiry_date_str, page_name)
                if calls is None or puts is None:
                    st.stop()
                
                if st.session_state.current_page == "Gamma Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S)
                elif st.session_state.current_page == "Vanna Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S)
                elif st.session_state.current_page == "Delta Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "DEX", "Delta Exposure by Strike", S)
                elif st.session_state.current_page == "Charm Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "Charm", "Charm Exposure by Strike", S)
                elif st.session_state.current_page == "Speed Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "Speed", "Speed Exposure by Strike", S)
                elif st.session_state.current_page == "Vomma Exposure":
                    fig_bar = create_exposure_bar_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S)
                
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{st.session_state.current_page}_bar_chart")

elif st.session_state.current_page == "Calculated Greeks":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        st.write("This page calculates delta, gamma, and vanna based on market data.")
        
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="calculated_greeks_ticker")
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, index=available_dates.index(saved_expiry_date) if saved_expiry_date in available_dates else 0, key="calculated_greeks_expiry_main")
                save_expiry_date(expiry_date_str)  # Save the expiry date
                calls, puts = fetch_options_for_date(ticker, expiry_date_str)
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                    st.stop()

                combined = pd.concat([calls, puts])
                combined = combined.dropna(subset=['extracted_expiry'])
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

                # Filter options by expiry
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

elif st.session_state.current_page == "Dashboard":
    main_container = st.container()
    with main_container:
        st.empty()
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="dashboard_ticker")
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, index=available_dates.index(saved_expiry_date) if saved_expiry_date in available_dates else 0, key="dashboard_expiry_main")
                save_expiry_date(expiry_date_str)  # Save the expiry date
                calls, puts, S, t, selected_expiry, today = compute_greeks_and_charts(ticker, expiry_date_str, "dashboard")
                if calls is None or puts is None:
                    st.stop()
                
                fig_gamma = create_exposure_bar_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S)
                fig_vanna = create_exposure_bar_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S)
                fig_delta = create_exposure_bar_chart(calls, puts, "DEX", "Delta Exposure by Strike", S)
                fig_charm = create_exposure_bar_chart(calls, puts, "Charm", "Charm Exposure by Strike", S)
                fig_speed = create_exposure_bar_chart(calls, puts, "Speed", "Speed Exposure by Strike", S)
                fig_vomma = create_exposure_bar_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S)
                
                # Intraday price chart
                intraday_data = stock.history(period="1d", interval="1m")
                if intraday_data.empty:
                    st.warning("No intraday data available for this ticker.")
                else:
                    # Initialize plot with cleared shapes/annotations
                    fig_intraday = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_intraday.layout.shapes = []
                    fig_intraday.layout.annotations = []
                    
                    # Add price trace
                    fig_intraday.add_trace(
                        go.Scatter(
                            x=intraday_data.index,
                            y=intraday_data['Close'],
                            name="Price",
                            line=dict(color='gold')
                        ),
                        secondary_y=False
                    )
                    
                    # Add current price annotation
                    current_price = intraday_data['Close'].iloc[-1]
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
                    
                    # Process options data
                    calls['OptionType'] = 'Call'
                    puts['OptionType'] = 'Put'
                    
                    # Combine and filter GEX data
                    options_df = pd.concat([calls, puts]).dropna(subset=['GEX'])
                    added_strikes = set()
                    
                    if options_df.empty:
                        st.warning("Intraday Data will display near market open.")
                    else:
                        top5 = options_df.nlargest(5, 'GEX')[['strike', 'GEX', 'OptionType']]
                        
                        # Find max GEX value for color scaling
                        max_gex = abs(top5['GEX']).max()
                        
                        # Add GEX levels
                        for row in top5.itertuples():
                            if row.strike not in added_strikes:
                                # Ensure intensity is not NaN and within valid range
                                if not pd.isna(row.GEX) and row.GEX != 0:
                                    # Modified color intensity calculation - now ranges from 0.4 to 1.0
                                    intensity = 0.4 + (min(abs(row.GEX) / max_gex, 1.0) * 0.6)
                                    
                                    # Ensure intensity is within valid range
                                    if not pd.isna(intensity) and 0 <= intensity <= 1:
                                        # Create RGB color based on option type and intensity
                                        if row.OptionType == 'Call':
                                            base_color = call_color
                                        else:
                                            base_color = put_color
                                            
                                        # Convert hex to RGB and apply intensity
                                        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                                        color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})'
                                        
                                        # Add horizontal line
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=row.strike,
                                            y1=row.strike,
                                            line=dict(color=color, width=2),
                                            xref='x',
                                            yref='y'
                                        )
                                        
                                        # Add GEX annotation with matching color
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
                    
                    # Update layout
                    fig_intraday.update_layout(
                        title=f"Intraday Price for {ticker}",
                        height=600,
                        hovermode='x unified',
                        margin=dict(r=150, l=50)
                    )
                    fig_intraday.update_xaxes(title_text="Time")
                    fig_intraday.update_yaxes(title_text="Price", secondary_y=False)

                
                # Calculate volume ratio
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                fig_volume_ratio = create_donut_chart(call_volume, put_volume)
                
                # Multi-select for choosing charts to display
                chart_options = ["Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Volume Ratio"]
                selected_charts = st.multiselect("Select charts to display:", chart_options, default=chart_options)
                
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
                
                # Display supplemental charts in rows of 2
                for i in range(0, len(supplemental_charts), 2):
                    cols = st.columns(2)
                    for j, chart in enumerate(supplemental_charts[i:i+2]):
                        cols[j].plotly_chart(chart, use_container_width=True)

# -----------------------------------------
# Auto-refresh
# -----------------------------------------
refresh_rate = 10  # in seconds
if not st.session_state.get("loading_complete", False):
    st.session_state.loading_complete = True
    st.rerun()
else:
    time.sleep(refresh_rate)
    st.rerun()

if not st.session_state.get("initialized", False):
    st.session_state.initialized = True
    st.rerun()
else:
    time.sleep(refresh_rate)
    st.rerun()