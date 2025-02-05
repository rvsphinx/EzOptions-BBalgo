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

def fetch_options_for_date(ticker, date):
    """
    Fetches option chains for the given ticker and date.
    Returns two DataFrames: one for calls and one for puts.
    """
    stock = yf.Ticker(ticker)
    try:
        chain = stock.option_chain(date)
        calls = chain.calls
        puts = chain.puts
        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options for date {date}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def fetch_data_with_cache(ticker, date, key_prefix):
    """Helper function to fetch data with caching"""
    current_time = time.time()
    last_fetch_time_key = f"{key_prefix}_last_fetch_time"
    data_key = f"{key_prefix}_data"
    last_ticker_key = f"{key_prefix}_last_ticker"
    last_date_key = f"{key_prefix}_last_date"
    
    if (last_fetch_time_key not in st.session_state or 
        current_time - st.session_state[last_fetch_time_key] > 30 or
        st.session_state.get(last_ticker_key) != ticker or
        st.session_state.get(last_date_key) != date):
        calls, puts = fetch_options_for_date(ticker, date)
        st.session_state[last_fetch_time_key] = current_time
        st.session_state[data_key] = (calls, puts)
        st.session_state[last_ticker_key] = ticker
        st.session_state[last_date_key] = date
    else:
        calls, puts = st.session_state[data_key]
    
    return calls, puts

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
# New function: Fetch all options and add extracted expiry column
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
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

# =========================================
# 2) Existing Visualization Functions
# =========================================
def create_oi_volume_charts(calls, puts):
    # Get underlying price
    stock = yf.Ticker(ticker)
    S = stock.info.get("regularMarketPrice")
    if S is None:
        S = stock.fast_info.get("lastPrice")
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    calls_df = calls[['strike', 'openInterest', 'volume']].copy()
    calls_df['OptionType'] = 'Call'
    
    puts_df = puts[['strike', 'openInterest', 'volume']].copy()
    puts_df['OptionType'] = 'Put'
    
    combined = pd.concat([calls_df, puts_df], ignore_index=True)
    combined.sort_values(by='strike', inplace=True)
    
    fig_oi = px.bar(
        combined,
        x='strike',
        y='openInterest',
        color='OptionType',
        title='Open Interest by Strike',
        barmode='group',
        color_discrete_map={'Call': 'green', 'Put': 'darkred'}  # Update colors
    )
    fig_oi.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        hovermode='x unified'
    )
    fig_oi.update_xaxes(rangeslider=dict(visible=True))
    
    fig_volume = px.bar(
        combined,
        x='strike',
        y='volume',
        color='OptionType',
        title='Volume by Strike',
        barmode='group',
        color_discrete_map={'Call': 'green', 'Put': 'darkred'}  # Update colors
    )
    fig_volume.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        hovermode='x unified'
    )
    fig_volume.update_xaxes(rangeslider=dict(visible=True))
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_oi, fig_volume

# Remove the create_heatmap function
# def create_heatmap(calls, puts, value='volume'):
#     # ...existing code...

def create_donut_chart(call_volume, put_volume):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        showlegend=True
    )
    fig.update_traces(hoverinfo='label+percent+value', marker=dict(colors=['green', 'darkred']))  # Update colors
    return fig

# =========================================
# 3) Greek Calculation Helper Function
# =========================================
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
        delta_val = bs_delta(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        vega_val = bs_vega(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        vanna_val = -vega_val * d2 / (S * sigma * sqrt(t))
        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

# Add charm calculation function
def calculate_charm(flag, S, K, t, sigma):
    """
    Calculate charm (dDelta/dTime) for an option.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        norm_d1 = norm.pdf(d1)
        norm_d2 = norm.pdf(d2)
        
        if flag == 'c':
            charm = -norm_d1 * (sigma / (2 * sqrt(t))) * (1 - d1/(sigma * sqrt(t)))
        else:  # put
            charm = -norm_d1 * (sigma / (2 * sqrt(t))) * (1 - d1/(sigma * sqrt(t)))
        
        return charm
    except Exception as e:
        st.error(f"Error calculating charm: {e}")
        return None

# Add error handling for fetching the last price to avoid KeyError.
def get_last_price(stock):
    """Helper function to get the last price of the stock."""
    try:
        S = stock.info.get("regularMarketPrice")
        if S is None:
            S = stock.fast_info.get("lastPrice")
        return S
    except KeyError:
        return None

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

# Add speed calculation function
def calculate_speed(flag, S, K, t, sigma):
    """
    Calculate speed (DgammaDspot) for an option.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        speed = -gamma_val * (1 + (d1 / (sigma * sqrt(t)))) / S
        return speed
    except Exception as e:
        st.error(f"Error calculating speed: {e}")
        return None

# =========================================
# 4) Streamlit App Navigation
# =========================================
st.title("Ez Options Stock Data")

def reset_session_state():
    """Reset all session state variables except for essential ones"""
    # Keep track of keys we want to preserve
    preserved_keys = {'current_page', 'initialized', 'saved_ticker'}  # added saved_ticker
    preserved_values = {key: st.session_state[key] 
                       for key in preserved_keys 
                       if key in st.session_state}
    
    # Clear everything safely
    for key in list(st.session_state.keys()):
        if key not in preserved_keys:
            try:
                del st.session_state[key]
            except KeyError:
                pass  # Ignore if key doesn't exist
    
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

def safe_rerun():
    """Helper function to handle rerun across different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.runtime.legacy_caching.clear_cache()
            st.empty()

# Add a function to save the ticker
def save_ticker(ticker):
    st.session_state.saved_ticker = ticker

def manual_refresh():
    """Manual refresh button to rerun the app"""
    if st.button("Refresh"):
        safe_rerun()

st.sidebar.title("Navigation")
pages = ["Dashboard", "Volume Ratio", "OI & Volume", "Gamma Exposure", "Delta Exposure", 
         "Vanna Exposure", "Charm Exposure", "Speed Exposure", "Calculated Greeks"]

new_page = st.sidebar.radio("Select a page:", pages)

if handle_page_change(new_page):
    safe_rerun()

# Use the saved ticker if available
saved_ticker = st.session_state.get("saved_ticker", "SPY")

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        # For future dates, ensure they're treated as valid
        current_market_date = datetime.now().date()
        # Return True if expiry is today or in the future
        return expiry_date >= current_market_date
    except Exception:
        return False

def compute_greeks_and_charts(ticker, expiry_date_str, page_key):
    calls, puts = fetch_data_with_cache(ticker, expiry_date_str, page_key)
    if calls.empty and puts.empty:
        st.warning("No options data available for this ticker.")
        return None, None, None, None, None, None

    combined = pd.concat([calls, puts])
    combined = combined.dropna(subset=['extracted_expiry'])
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    calls = calls[calls['extracted_expiry'] == selected_expiry]
    puts = puts[puts['extracted_expiry'] == selected_expiry]

    stock = yf.Ticker(ticker)
    S = get_last_price(stock)
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

    # Compute Greeks for Gamma, Vanna, Delta, Charm, and Speed
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

    calls = calls.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed"])
    puts = puts.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed"])

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

    return calls, puts, S, t, selected_expiry, today

def create_exposure_bar_chart(calls, puts, exposure_type, title, S):
    # Filter out zero values
    calls_df = calls[['strike', exposure_type]].copy()
    calls_df = calls_df[calls_df[exposure_type] != 0]
    calls_df['OptionType'] = 'Call'

    puts_df = puts[['strike', exposure_type]].copy()
    puts_df = puts_df[puts_df[exposure_type] != 0]
    puts_df['OptionType'] = 'Put'

    combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
    combined_chart.sort_values(by='strike', inplace=True)

    fig = px.bar(
        combined_chart,
        x='strike',
        y=exposure_type,
        color='OptionType',
        title=title,
        barmode='group',
        color_discrete_map={'Call': 'green', 'Put': 'darkred'}
    )

    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title=title,
        hovermode='x unified'
    )

    fig.update_xaxes(rangeslider=dict(visible=True))

    # Zoom into the range around the current price
    zoom_range = 0.1 * S  # 10% of the current price
    min_strike = max(S - zoom_range, combined_chart['strike'].min())
    max_strike = min(S + zoom_range, combined_chart['strike'].max())
    fig.update_xaxes(range=[min_strike, max_strike])

    fig = add_current_price_line(fig, S)

    return fig

if st.session_state.current_page == "OI & Volume":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        manual_refresh()  # Add refresh button
        st.write("**Select filters below to see updated data, charts, and tables.**")
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="options_data_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates)
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "options_data")
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
                        min_strike = float(min(calls['strike'].min(), puts['strike'].min()))
                        max_strike = float(max(calls['strike'].max(), puts['strike'].max()))
                        strike_range = st.slider(
                            "Select Strike Range:",
                            min_value=min_strike,
                            max_value=max_strike,
                            value=(min_strike, max_strike),
                            step=1.0
                        )
                        volume_over_oi = st.checkbox("Show only rows where Volume > Open Interest")
                        min_selected, max_selected = strike_range
                        calls_filtered = calls[(calls['strike'] >= min_selected) & (calls['strike'] <= max_selected)].copy()
                        puts_filtered = puts[(puts['strike'] >= min_selected) & (puts['strike'] <= max_selected)].copy()
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
        manual_refresh()  # Add refresh button
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="volume_ratio_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key="volume_ratio_expiry_main")
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "volume_ratio")
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

elif st.session_state.current_page in ["Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure", "Speed Exposure"]:
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        manual_refresh()  # Add refresh button
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, or speed
        user_ticker = st.text_input(
            "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", 
            saved_ticker, 
            key=f"{page_name}_exposure_ticker"
        )
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key=f"{page_name}_expiry_main")
                calls, puts, S, t, selected_expiry, today = compute_greeks_and_charts(ticker, expiry_date_str, f"{page_name}_exposure")
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
                
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{st.session_state.current_page}_bar_chart")

elif st.session_state.current_page == "Calculated Greeks":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        manual_refresh()  # Add refresh button
        st.write("This page calculates delta, gamma, and vanna based on market data.")
        
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="calculated_greeks_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key="calculated_greeks_expiry_main")
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "calculated_greeks")
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
        manual_refresh()  # Add refresh button
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="dashboard_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key="dashboard_expiry_main")
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "dashboard")
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                else:
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    
                    # Get underlying price
                    stock = yf.Ticker(ticker)
                    S = get_last_price(stock)
                    if S is None:
                        st.error("Could not fetch underlying price.")
                    else:
                        S = round(S, 2)
                        st.markdown(f"**Underlying Price (S):** {S}")
                        today = datetime.today().date()
                        t_days = (selected_expiry - today).days
                        if not is_valid_trading_day(selected_expiry, today):
                            st.error("The selected expiration date is in the past!")
                        else:
                            t = t_days / 365.0
                            st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                            
                            # Compute Greeks for Gamma, Vanna, Delta, and Charm
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
                            puts["calc_speed"] = puts.apply(lambda row: compute_speed(row, "p"), axis=1)
                            calls["calc_speed"] = calls.apply(lambda row: compute_speed(row, "c"), axis=1)
                            
                            calls = calls.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed"])
                            puts = puts.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed"])
                            
                            calls["GEX"] = calls["calc_gamma"] * calls["openInterest"] * 100
                            puts["GEX"] = puts["calc_gamma"] * puts["openInterest"] * 100
                            calls["VEX"] = calls["calc_vanna"] * calls["openInterest"] * 100
                            puts["VEX"] = puts["calc_vanna"] * puts["openInterest"] * 100
                            calls["DEX"] = calls["calc_delta"] * calls["openInterest"] * 100
                            puts["DEX"] = puts["calc_delta"] * puts["openInterest"] * 100
                            calls["Charm"] = calls["calc_charm"] * calls["openInterest"] * 100
                            puts["Charm"] = puts["calc_charm"] * puts["openInterest"] * 100
                            puts["Speed"] = puts["calc_speed"] * puts["openInterest"] * 100
                            calls["Speed"] = calls["calc_speed"] * calls["openInterest"] * 100
                            
                            # Create bar charts for Gamma, Vanna, Delta, and Charm
                            def create_exposure_bar_chart(calls, puts, exposure_type, title):
                                # Filter out zero values
                                calls_df = calls[['strike', exposure_type]].copy()
                                calls_df = calls_df[calls_df[exposure_type] != 0]
                                calls_df['OptionType'] = 'Call'
                                
                                puts_df = puts[['strike', exposure_type]].copy()
                                puts_df = puts_df[puts_df[exposure_type] != 0]
                                puts_df['OptionType'] = 'Put'
                                
                                combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                                combined_chart.sort_values(by='strike', inplace=True)
                                
                                fig = px.bar(
                                    combined_chart,
                                    x='strike',
                                    y=exposure_type,
                                    color='OptionType',
                                    title=title,
                                    barmode='group',
                                    color_discrete_map={'Call': 'green', 'Put': 'darkred'}
                                )
                                
                                fig.update_layout(
                                    xaxis_title='Strike Price',
                                    yaxis_title=title,
                                    hovermode='x unified'
                                )
                                
                                fig.update_xaxes(rangeslider=dict(visible=True))
                                
                                # Zoom into the range around the current price
                                zoom_range = 0.1 * S  # 10% of the current price
                                min_strike = max(S - zoom_range, combined_chart['strike'].min())
                                max_strike = min(S + zoom_range, combined_chart['strike'].max())
                                fig.update_xaxes(range=[min_strike, max_strike])
                                
                                fig = add_current_price_line(fig, S)
                                
                                return fig
                            
                            fig_gamma = create_exposure_bar_chart(calls, puts, "GEX", "Gamma Exposure by Strike")
                            fig_vanna = create_exposure_bar_chart(calls, puts, "VEX", "Vanna Exposure by Strike")
                            fig_delta = create_exposure_bar_chart(calls, puts, "DEX", "Delta Exposure by Strike")
                            fig_charm = create_exposure_bar_chart(calls, puts, "Charm", "Charm Exposure by Strike")
                            fig_speed = create_exposure_bar_chart(calls, puts, "Speed", "Speed Exposure by Strike")
                            
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
                                                        color = f'rgba(0, {int(255 * intensity)}, 0, 0.9)'  # Increased opacity to 0.9
                                                    else:
                                                        color = f'rgba({int(255 * intensity)}, 0, 0, 0.9)'  # Increased opacity to 0.9
                                                    
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
                            chart_options = ["Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure", "Speed Exposure", "Volume Ratio"]
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
                            if "Volume Ratio" in selected_charts:
                                supplemental_charts.append(fig_volume_ratio)
                            
                            # Display supplemental charts in rows of 2
                            for i in range(0, len(supplemental_charts), 2):
                                cols = st.columns(2)
                                for j, chart in enumerate(supplemental_charts[i:i+2]):
                                    cols[j].plotly_chart(chart, use_container_width=True)

elif st.session_state.current_page == "Charm Exposure":
    main_container = st.container()
    with main_container:
        st.empty()
        manual_refresh()  # Add refresh button
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="charm_exposure_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key="charm_expiry_main")
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "charm_exposure")
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                else:
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    
                    # Get underlying price
                    stock = yf.Ticker(ticker)
                    S = get_last_price(stock)
                    if S is None:
                        st.error("Could not fetch underlying price.")
                    else:
                        S = round(S, 2)
                        st.markdown(f"**Underlying Price (S):** {S}")
                        today = datetime.today().date()
                        t_days = (selected_expiry - today).days
                        if not is_valid_trading_day(selected_expiry, today):
                            st.error("The selected expiration date is in the past!")
                        else:
                            t = t_days / 365.0
                            st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                            
                            def compute_charm(row, flag):
                                sigma = row.get("impliedVolatility", None)
                                if sigma is None or sigma <= 0:
                                    return None
                                try:
                                    charm_val = calculate_charm(flag, S, row["strike"], t, sigma)
                                    return charm_val
                                except Exception:
                                    return None
                            
                            calls = calls.copy()
                            puts = puts.copy()
                            calls["calc_charm"] = calls.apply(lambda row: compute_charm(row, "c"), axis=1)
                            puts["calc_charm"] = puts.apply(lambda row: compute_charm(row, "p"), axis=1)
                            
                            calls = calls.dropna(subset=["calc_charm"])
                            puts = puts.dropna(subset=["calc_charm"])
                            
                            calls["Charm"] = calls["calc_charm"] * calls["openInterest"] * 100
                            puts["Charm"] = puts["calc_charm"] * puts["openInterest"] * 100
                            
                            def create_charm_bar_chart(calls, puts):
                                calls_df = calls[['strike', 'Charm']].copy()
                                calls_df['OptionType'] = 'Call'
                                puts_df = puts[['strike', 'Charm']].copy()
                                puts_df['OptionType'] = 'Put'
                                combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                                combined_chart = combined_chart[combined_chart['Charm'] != 0]  # Exclude values of 0
                                combined_chart.sort_values(by='strike', inplace=True)
                                
                                fig = px.bar(
                                    combined_chart,
                                    x='strike',
                                    y='Charm',
                                    color='OptionType',
                                    title='Charm Exposure by Strike',
                                    barmode='group',
                                    color_discrete_map={'Call': 'green', 'Put': 'darkred'}  # Update colors
                                )
                                fig.update_layout(
                                    xaxis_title='Strike Price',
                                    yaxis_title='Charm Exposure',
                                    hovermode='x unified'
                                )
                                fig.update_xaxes(rangeslider=dict(visible=True))
                                
                                # Zoom into the range around the current price
                                zoom_range = 0.1 * S  # 10% of the current price
                                min_strike = max(S - zoom_range, combined_chart['strike'].min())
                                max_strike = min(S + zoom_range, combined_chart['strike'].max())
                                
                                fig.update_xaxes(range=[min_strike, max_strike])
                                
                                fig = add_current_price_line(fig, S)  # Add price line
                                
                                return fig
                            
                            fig_bar = create_charm_bar_chart(calls, puts)
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"Charm Exposure_bar_chart")

elif st.session_state.current_page == "Speed Exposure":
    main_container = st.container()
    with main_container:
        st.empty()
        manual_refresh()  # Add refresh button
        user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="speed_exposure_ticker")
        ticker = format_ticker(user_ticker)
        save_ticker(user_ticker)  # Save the ticker
        
        if ticker:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                expiry_date_str = st.selectbox("Select an Exp. Date:", options=available_dates, key="speed_expiry_main")
                calls, puts = fetch_data_with_cache(ticker, expiry_date_str, "speed_exposure")
                if calls.empty and puts.empty:
                    st.warning("No options data available for this ticker.")
                else:
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    
                    # Get underlying price
                    stock = yf.Ticker(ticker)
                    S = get_last_price(stock)
                    if S is None:
                        st.error("Could not fetch underlying price.")
                    else:
                        S = round(S, 2)
                        st.markdown(f"**Underlying Price (S):** {S}")
                        today = datetime.today().date()
                        t_days = (selected_expiry - today).days
                        if not is_valid_trading_day(selected_expiry, today):
                            st.error("The selected expiration date is in the past!")
                        else:
                            t = t_days / 365.0
                            st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                            
                            def compute_speed(row, flag):
                                sigma = row.get("impliedVolatility", None)
                                if sigma is None or sigma <= 0:
                                    return None
                                try:
                                    speed_val = calculate_speed(flag, S, row["strike"], t, sigma)
                                    return speed_val
                                except Exception:
                                    return None
                            
                            calls = calls.copy()
                            puts = puts.copy()
                            calls["calc_speed"] = calls.apply(lambda row: compute_speed(row, "c"), axis=1)
                            puts["calc_speed"] = puts.apply(lambda row: compute_speed(row, "p"), axis=1)
                            
                            calls = calls.dropna(subset=["calc_speed"])
                            puts = puts.dropna(subset=["calc_speed"])
                            
                            calls["Speed"] = calls["calc_speed"] * calls["openInterest"] * 100
                            puts["Speed"] = puts["calc_speed"] * puts["openInterest"] * 100
                            
                            def create_speed_bar_chart(calls, puts):
                                calls_df = calls[['strike', 'Speed']].copy()
                                calls_df['OptionType'] = 'Call'
                                puts_df = puts[['strike', 'Speed']].copy()
                                puts_df['OptionType'] = 'Put'
                                combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                                combined_chart = combined_chart[combined_chart['Speed'] != 0]  # Exclude values of 0
                                combined_chart.sort_values(by='strike', inplace=True)
                                
                                fig = px.bar(
                                    combined_chart,
                                    x='strike',
                                    y='Speed',
                                    color='OptionType',
                                    title='Speed Exposure by Strike',
                                    barmode='group',
                                    color_discrete_map={'Call': 'green', 'Put': 'darkred'}  # Update colors
                                )
                                fig.update_layout(
                                    xaxis_title='Strike Price',
                                    yaxis_title='Speed Exposure',
                                    hovermode='x unified'
                                )
                                fig.update_xaxes(rangeslider=dict(visible=True))
                                
                                # Zoom into the range around the current price
                                zoom_range = 0.1 * S  # 10% of the current price
                                min_strike = max(S - zoom_range, combined_chart['strike'].min())
                                max_strike = min(S + zoom_range, combined_chart['strike'].max())
                                
                                fig.update_xaxes(range=[min_strike, max_strike])
                                
                                fig = add_current_price_line(fig, S)  # Add price line
                                
                                return fig
                            
                            fig_bar = create_speed_bar_chart(calls, puts)
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"Speed Exposure_bar_chart")

# -----------------------------------------
# Auto-refresh
# -----------------------------------------
refresh_rate = 10  # in seconds
if not st.session_state.get("loading_complete", False):
    st.session_state.loading_complete = True
    safe_rerun()
else:
    time.sleep(refresh_rate)
    safe_rerun()

# Remove redundant initialization check
if not st.session_state.get("initialized", False):
    st.session_state.initialized = True
    safe_rerun()