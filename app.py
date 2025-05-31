import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="V-Macro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SignalColor(Enum):
    GREEN = (1, "Good", "#00FF00")
    YELLOW = (0, "Uncertainty", "#FFFF00") 
    ORANGE = (-0.5, "Caution", "#FFA500")
    RED = (-1, "Bad", "#FF0000")

@dataclass
class IndicatorConfig:
    name: str
    category: str
    fred_series_id: str = ""
    signal_logic: str = "higher_is_better"

class DataFetcher:
    def __init__(self, fred_api_key: str):
        self.fred_api_key = fred_api_key
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
    def fetch_fred_data(self, series_id: str, limit: int = 100) -> Optional[pd.DataFrame]:
        if not self.fred_api_key or not series_id:
            return None
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'limit': limit,
            'sort_order': 'desc'
        }
        
        try:
            response = requests.get(self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.dropna().sort_values('date')
                
        except Exception as e:
            st.error(f"Error fetching data for {series_id}: {e}")
            
        return None

class SignalGenerator:
    def __init__(self):
        self.lookback_periods = {
            '< 3 months': 60,
            '3-6 months': 120,
            '6-12 months': 250,
            '12-24 months': 500
        }
    
    def calculate_signals(self, data: pd.DataFrame, signal_logic: str) -> Dict[str, SignalColor]:
        if data is None or len(data) < 10:
            return {period: SignalColor.YELLOW for period in self.lookback_periods.keys()}
            
        signals = {}
        current_value = data['value'].iloc[-1]
        
        for period, days in self.lookback_periods.items():
            try:
                if len(data) > days // 7:
                    historical_value = data['value'].iloc[-(days // 7)]
                else:
                    historical_value = data['value'].iloc[0]
                
                pct_change = (current_value - historical_value) / abs(historical_value) * 100
                
                if signal_logic == "higher_is_better":
                    if pct_change > 5:
                        signals[period] = SignalColor.GREEN
                    elif pct_change > 0:
                        signals[period] = SignalColor.YELLOW
                    elif pct_change > -5:
                        signals[period] = SignalColor.ORANGE
                    else:
                        signals[period] = SignalColor.RED
                else:
                    if pct_change < -5:
                        signals[period] = SignalColor.GREEN
                    elif pct_change < 0:
                        signals[period] = SignalColor.YELLOW
                    elif pct_change < 5:
                        signals[period] = SignalColor.ORANGE
                    else:
                        signals[period] = SignalColor.RED
                        
            except Exception:
                signals[period] = SignalColor.YELLOW
                
        return signals

def get_sample_indicators():
    """Sample indicators for demonstration"""
    return [
        IndicatorConfig("Federal Funds Rate", "Policy", "FEDFUNDS", "lower_is_better"),
        IndicatorConfig("Unemployment Rate", "Economy", "UNRATE", "lower_is_better"),
        IndicatorConfig("Consumer Price Index", "Economy", "CPIAUCSL", "lower_is_better"),
        IndicatorConfig("Industrial Production", "Economy", "INDPRO", "higher_is_better"),
        IndicatorConfig("Consumer Sentiment", "Economy", "UMCSENT", "higher_is_better"),
        IndicatorConfig("10-Year Treasury Rate", "Financial Markets", "GS10", "lower_is_better"),
        IndicatorConfig("VIX Volatility Index", "Market Structure", "VIXCLS", "lower_is_better"),
        IndicatorConfig("Real GDP", "Economy", "GDPC1", "higher_is_better"),
    ]

def main():
    st.title("📊 V-Macro Dashboard")
    st.markdown("**Real-time Economic Indicator Tracking & Signal Generation**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    fred_api_key = st.secrets.get("FRED_API_KEY", "")
    
    if not fred_api_key:
        st.error("🔑 FRED API Key not configured. Please add it to your Streamlit secrets.")
        st.info("Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Initialize components
    data_fetcher = DataFetcher(fred_api_key)
    signal_generator = SignalGenerator()
    
    # Load sample indicators
    indicators = get_sample_indicators()
    
    # Time horizon selection
    time_horizons = ['< 3 months', '3-6 months', '6-12 months', '12-24 months']
    selected_horizon = st.sidebar.selectbox("📅 Select Time Horizon", time_horizons)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main dashboard
    st.header(f"Dashboard - {selected_horizon}")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch data and calculate signals
    dashboard_data = []
    categories = {}
    
    for i, indicator in enumerate(indicators):
        progress_bar.progress((i + 1) / len(indicators))
        status_text.text(f"Processing: {indicator.name}")
        
        # Fetch data
        data = data_fetcher.fetch_fred_data(indicator.fred_series_id)
        
        # Calculate signals
        signals = signal_generator.calculate_signals(data, indicator.signal_logic)
        current_signal = signals.get(selected_horizon, SignalColor.YELLOW)
        
        # Store for category aggregation
        if indicator.category not in categories:
            categories[indicator.category] = []
        categories[indicator.category].append(current_signal.value[0])
        
        # Add to dashboard data
        dashboard_data.append({
            'Indicator': indicator.name,
            'Category': indicator.category,
            'Signal': current_signal.value[1],
            'Color': current_signal.value[2],
            'Score': current_signal.value[0]
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Create dashboard DataFrame
    df = pd.DataFrame(dashboard_data)
    
    # Category summary
    st.subheader("📈 Category Overview")
    
    category_summary = []
    for category, scores in categories.items():
        avg_score = np.mean(scores)
        if avg_score > 0.5:
            signal = SignalColor.GREEN
        elif avg_score > 0:
            signal = SignalColor.YELLOW
        elif avg_score > -0.5:
            signal = SignalColor.ORANGE
        else:
            signal = SignalColor.RED
            
        category_summary.append({
            'Category': category,
            'Signal': signal.value[1],
            'Score': avg_score
        })
    
    category_df = pd.DataFrame(category_summary)
    
    # Display category overview
    cols = st.columns(len(category_df))
    for i, (_, row) in enumerate(category_df.iterrows()):
        with cols[i]:
            if row['Signal'] == 'Good':
                st.success(f"**{row['Category']}**\n{row['Signal']}")
            elif row['Signal'] == 'Bad':
                st.error(f"**{row['Category']}**\n{row['Signal']}")
            elif row['Signal'] == 'Caution':
                st.warning(f"**{row['Category']}**\n{row['Signal']}")
            else:
                st.info(f"**{row['Category']}**\n{row['Signal']}")
    
    # Detailed indicators table
    st.subheader("📊 Detailed Indicators")
    
    # Color-code the dataframe
    def color_signals(val):
        if val == 'Good':
            return 'background-color: #90EE90'
        elif val == 'Bad':
            return 'background-color: #FFB6C1'
        elif val == 'Caution':
            return 'background-color: #FFE4B5'
        else:
            return 'background-color: #FFFFE0'
    
    styled_df = df[['Indicator', 'Category', 'Signal']].style.applymap(
        color_signals, subset=['Signal']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Charts
    st.subheader("📈 Signal Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal distribution pie chart
        signal_counts = df['Signal'].value_counts()
        fig_pie = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Overall Signal Distribution",
            color_discrete_map={
                'Good': '#00FF00',
                'Uncertainty': '#FFFF00',
                'Caution': '#FFA500',
                'Bad': '#FF0000'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Category breakdown
        category_counts = df.groupby(['Category', 'Signal']).size().reset_index(name='Count')
        fig_bar = px.bar(
            category_counts,
            x='Category',
            y='Count',
            color='Signal',
            title="Signals by Category",
            color_discrete_map={
                'Good': '#00FF00',
                'Uncertainty': '#FFFF00',
                'Caution': '#FFA500',
                'Bad': '#FF0000'
            }
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source:** Federal Reserve Economic Data (FRED)")

if __name__ == "__main__":
    main()
