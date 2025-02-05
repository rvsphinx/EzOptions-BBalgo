# Ez Options

A real-time options analysis tool that provides interactive visualizations of options data, greeks, and market indicators.

![Ez Options Dashboard](https://i.imgur.com/8hT2LZ4.png)

## Quick Start

### Windows Users:
1. Run the requirements installer:
   ```bat
   requirements.bat
   ```
2. Launch the dashboard:
   ```bat
   ezoptions.bat
   ```

### Mac Users:
1. Manually install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the script in the terminal:
   ```sh
   python -m streamlit run https://raw.githubusercontent.com/EazyDuz1t/ezoptions/refs/heads/main/ezoptions.py
   ```

## Features

- Real-time options data visualization
- Interactive dashboard with multiple views:
  - Volume and Open Interest analysis
  - Greeks exposure (Delta, Gamma, Vanna, Charm)
  - Intraday price tracking with key levels

## Requirements

- Python 3.x

Required Python packages will be automatically installed by the `requirements.bat` file on Windows or manually installed using `pip install -r requirements.txt` on Mac.

## Usage

After launching, enter any stock ticker (e.g., AAPL, TSLA, SPX) to analyze its options data. Navigate between different views using the sidebar menu.

## Credits

Based on: https://github.com/anvgun/Options_Analyzer
