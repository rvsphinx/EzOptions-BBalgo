# Ez Options

A real-time options analysis tool that provides interactive visualizations of options data, greeks, and market indicators.

![Ez Options Dashboard](https://i.imgur.com/8hT2LZ4.png)

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Credits](#credits)

## Quick Start

1.  Clone the repository: `git clone https://github.com/EazyDuz1t/ezoptions.git`
2.  Navigate to the project directory: `cd ezoptions`
3.  Create a virtual environment: `python -m venv venv`
4.  Activate the virtual environment:
    *   On Windows: `.\venv\Scripts\activate`
    *   On macOS and Linux: `source venv/bin/activate`
5.  Run `python main.py` to install dependencies and launch the dashboard using Streamlit.

## Features

-   **Real-time options data visualization:** Provides up-to-the-minute options chain data, allowing users to see the latest prices, volumes, and open interest for calls and puts.

-   **Interactive dashboard with multiple views:** A user-friendly interface with different panels for analyzing options data:

    *   **Volume and Open Interest analysis:** Visualize volume and open interest data to identify potential support and resistance levels.
    *   **Greeks exposure (Delta, Gamma, Vanna, Charm):** Calculate and display the Greeks for a given option, providing insights into its sensitivity to changes in price, time, and volatility.
    *   **Intraday price tracking with key levels:** Track the intraday price movements of the underlying asset and identify key support and resistance levels.

## Requirements

-   **Python:** 3.8 or higher

-   **Dependencies:** Required Python packages are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`.

## Usage

1.  **Launch the dashboard:** Follow the instructions in the [Quick Start](#quick-start) section to launch the dashboard.

2.  **Enter a stock ticker:** In the input field, enter the ticker symbol of the stock you want to analyze (e.g., AAPL for Apple, TSLA for Tesla, SPX for S\&P 500).

3.  **Explore the dashboard:** Use the sidebar menu to navigate between different views:

    *   **Volume and Open Interest:** Analyze the volume and open interest data to identify potential support and resistance levels. Look for large spikes in volume or open interest at specific strike prices, which may indicate significant levels.
    *   **Greeks Exposure:** Examine the Greeks (Delta, Gamma, Vanna, Charm) to understand the option's sensitivity to changes in price, time, and volatility. Use this information to assess the risk and potential reward of the option.
    *   **Intraday Price Tracking:** Track the intraday price movements of the underlying asset and identify key support and resistance levels. Look for patterns such as double tops, double bottoms, and trendlines.

4.  **Perform analysis:** Use the interactive tools and visualizations to perform your own analysis. For example, you can:

    *   Identify potential trading opportunities based on volume and open interest data.
    *   Assess the risk of an option position based on its Greeks.
    *   Track the price movements of the underlying asset to identify potential entry and exit points.

## How to Keep Updated

1.  Navigate to the project directory: `cd ezoptions`
2.  Run `git pull` to update the project to the latest version.

## Credits

-   Based on the Options Analyzer project by [anvgun](https://github.com/anvgun/Options_Analyzer).

-   Additional contributions by [EazyDuz1t](https://github.com/EazyDuz1t).
