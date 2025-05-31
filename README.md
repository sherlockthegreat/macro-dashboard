# V-Macro Dashboard

A comprehensive macro-economic indicator tracking dashboard built with Streamlit, featuring real-time data from the Federal Reserve Economic Data (FRED) API.

## 🚀 Live Dashboard

[Your Streamlit App URL will appear here after deployment]

## ✨ Features

- **Real-time Economic Data**: Fetches latest data from FRED API
- **Multi-timeframe Analysis**: Signals across 4 time horizons (< 3 months, 3-6 months, 6-12 months, 12-24 months)
- **Color-coded Signals**: Visual indicators for Good, Caution, Uncertainty, and Bad conditions
- **Interactive Charts**: Plotly-powered visualizations
- **Data Export**: Comprehensive CSV export functionality
- **Backend Monitoring**: API performance tracking and data quality metrics
- **Dark Mode Compatible**: Enhanced visibility in all themes

## 📊 Tracked Indicators

### Economy
- Federal Funds Rate
- Unemployment Rate
- Consumer Price Index
- Industrial Production
- Consumer Sentiment
- Real GDP

### Financial Markets
- 10-Year Treasury Rate
- VIX Volatility Index
- S&P 500

### Currency
- Dollar Index

## 🛠️ Setup Instructions

1. **Get FRED API Key**
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Create free account and generate API key

2. **Deploy to Streamlit Community Cloud**
   - Fork this repository
   - Connect to share.streamlit.io
   - Add `FRED_API_KEY` to secrets in deployment settings

3. **Local Development**
