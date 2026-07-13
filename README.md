# ◈ Stock Oracle

![License](https://img.shields.io/github/license/maupatel/stock-oracle)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)

Browser-based multi-factor stock analysis and discovery platform. One file, one command, full workbench.

## Features

- **Multi-factor analysis**: trend (SMA, EMA, MACD), momentum (RSI), volatility (Bollinger, ATR), and volume (OBV) indicators combined into a single view
- **Discovery**: "Top Opportunities" screener mode for finding candidates, plus a personal watchlist
- **News sentiment**: VADER sentiment scoring on live headlines
- **Interactive charts**: Plotly candlesticks and overlays with selectable ranges
- **Light and dark themes**

## Run it

```bash
pip install -r requirements.txt
streamlit run oracle_app.py
```

See [DEPLOY.md](DEPLOY.md) for hosting options.

## Stack

Python, Streamlit, yfinance, pandas, Plotly, ta, VADER.

Built by [Maulik Patel](https://github.com/maupatel), Senior Data Analyst at Capital One.
