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
        self.fetch_log = []  # Track all API calls
        
    def fetch_fred_data(self, series_id: str, limit: int = 100) -> Optional[pd.DataFrame]:
        if not self.fred_api_key or not series_id:
            self.fetch_log.append({
                'series_id': series_id,
                'status': 'Failed - No API key or series ID',
                'timestamp': dt.datetime.now(),
                'data_points': 0
            })
            return None
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'limit': limit,
            'sort_order': 'desc'
        }
        
        try:
            start_time = time.time()
            response = requests.get(self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                clean_df = df.dropna().sort_values('date')
                
                # Log successful fetch
                self.fetch_log.append({
                    'series_id': series_id,
                    'status': 'Success',
                    'timestamp': dt.datetime.now(),
                    'data_points': len(clean_df),
                    'latest_date': clean_df['date'].max() if len(clean_df) > 0 else None,
                    'response_time': round(time.time() - start_time, 2)
                })
                
                return clean_df
            else:
                self.fetch_log.append({
                    'series_id': series_id,
                    'status': 'Failed - No observations in response',
                    'timestamp': dt.datetime.now(),
                    'data_points': 0
                })
                
        except Exception as e:
            self.fetch_log.append({
                'series_id': series_id,
                'status': f'Failed - {str(e)}',
                'timestamp': dt.datetime.now(),
                'data_points': 0
            })
            
        return None

class SignalGenerator:
    def __init__(self):
        self.lookback_periods = {
            '< 3 months': 60,
            '3-6 months': 120,
            '6-12 months': 250,
            '12-24 months': 500
        }
        self.signal_log = []  # Track signal calculations
    
    def calculate_signals(self, data: pd.DataFrame, signal_logic: str, indicator_name: str) -> Dict[str, SignalColor]:
        if data is None or len(data) < 10:
            self.signal_log.append({
                'indicator': indicator_name,
                'status': 'Insufficient data',
                'data_points': len(data) if data is not None else 0,
                'signals': {period: 'YELLOW' for period in self.lookback_periods.keys()}
            })
            return {period: SignalColor.YELLOW for period in self.lookback_periods.keys()}
            
        signals = {}
        current_value = data['value'].iloc[-1]
        signal_details = {}
        
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
                
                signal_details[period] = {
                    'current_value': current_value,
                    'historical_value': historical_value,
                    'pct_change': round(pct_change, 2),
                    'signal': signals[period].value[1]
                }
                        
            except Exception as e:
                signals[period] = SignalColor.YELLOW
                signal_details[period] = {'error': str(e)}
        
        # Log signal calculation
        self.signal_log.append({
            'indicator': indicator_name,
            'status': 'Success',
            'data_points': len(data),
            'latest_date': data['date'].max(),
            'signal_details': signal_details
        })
                
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
    
    # Sidebar options
    time_horizons = ['< 3 months', '3-6 months', '6-12 months', '12-24 months']
    selected_horizon = st.sidebar.selectbox("📅 Select Time Horizon", time_horizons)
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False)
    show_raw_data = st.sidebar.checkbox("📋 Show Raw Data", value=False)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Backend Data", "📈 Raw Data", "⚙️ System Health"])
    
    with tab1:
        # Main dashboard
        st.header(f"Dashboard - {selected_horizon}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data and calculate signals
        dashboard_data = []
        categories = {}
        raw_data_store = {}
        
        for i, indicator in enumerate(indicators):
            progress_bar.progress((i + 1) / len(indicators))
            status_text.text(f"Processing: {indicator.name}")
            
            # Fetch data
            data = data_fetcher.fetch_fred_data(indicator.fred_series_id)
            raw_data_store[indicator.name] = data
            
            # Calculate signals
            signals = signal_generator.calculate_signals(data, indicator.signal_logic, indicator.name)
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
                'Score': current_signal.value[0],
                'Data Points': len(data) if data is not None else 0,
                'Latest Date': data['date'].max().strftime('%Y-%m-%d') if data is not None and len(data) > 0 else 'No Data'
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
        
        styled_df = df[['Indicator', 'Category', 'Signal', 'Data Points', 'Latest Date']].style.applymap(
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
    
    with tab2:
        # Backend Data Analysis
        st.header("🔍 Backend Data & Signal Analysis")
        
        # API Fetch Log
        st.subheader("📡 API Fetch Log")
        if data_fetcher.fetch_log:
            fetch_df = pd.DataFrame(data_fetcher.fetch_log)
            st.dataframe(fetch_df, use_container_width=True)
            
            # Success rate
            success_rate = len(fetch_df[fetch_df['status'] == 'Success']) / len(fetch_df) * 100
            st.metric("API Success Rate", f"{success_rate:.1f}%")
        
        # Signal Calculation Details
        st.subheader("🎯 Signal Calculation Details")
        if signal_generator.signal_log:
            for log_entry in signal_generator.signal_log:
                with st.expander(f"📊 {log_entry['indicator']} - {log_entry['status']}"):
                    st.write(f"**Data Points:** {log_entry['data_points']}")
                    if 'latest_date' in log_entry:
                        st.write(f"**Latest Date:** {log_entry['latest_date']}")
                    
                    if 'signal_details' in log_entry:
                        details_df = pd.DataFrame.from_dict(log_entry['signal_details'], orient='index')
                        st.dataframe(details_df)
        
        # Data Quality Metrics
        st.subheader("📈 Data Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_indicators = len(indicators)
            st.metric("Total Indicators", total_indicators)
        
        with col2:
            indicators_with_data = len([d for d in dashboard_data if d['Data Points'] > 0])
            st.metric("Indicators with Data", indicators_with_data)
        
        with col3:
            data_coverage = indicators_with_data / total_indicators * 100 if total_indicators > 0 else 0
            st.metric("Data Coverage", f"{data_coverage:.1f}%")
    
    with tab3:
        # Raw Data Viewer
        st.header("📋 Raw Economic Data")
        
        selected_indicator = st.selectbox(
            "Select Indicator to View Raw Data:",
            [ind.name for ind in indicators]
        )
        
        if selected_indicator in raw_data_store and raw_data_store[selected_indicator] is not None:
            raw_data = raw_data_store[selected_indicator]
            
            st.subheader(f"Raw Data: {selected_indicator}")
            st.write(f"**Data Points:** {len(raw_data)}")
            st.write(f"**Date Range:** {raw_data['date'].min()} to {raw_data['date'].max()}")
            st.write(f"**Latest Value:** {raw_data['value'].iloc[-1]}")
            
            # Show raw data table
            st.dataframe(raw_data.tail(50), use_container_width=True)
            
            # Plot time series
            fig = px.line(
                raw_data.tail(100), 
                x='date', 
                y='value', 
                title=f"{selected_indicator} - Time Series"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = raw_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Raw Data as CSV",
                data=csv,
                file_name=f"{selected_indicator.replace(' ', '_')}_data.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"No data available for {selected_indicator}")
    
    with tab4:
        # System Health
        st.header("⚙️ System Health & Performance")
        
        # API Status
        st.subheader("🌐 API Status")
        try:
            test_response = requests.get(
                "https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=" + fred_api_key + "&file_type=json",
                timeout=5
            )
            if test_response.status_code == 200:
                st.success("✅ FRED API Connection: Healthy")
            else:
                st.error(f"❌ FRED API Connection: Error {test_response.status_code}")
        except Exception as e:
            st.error(f"❌ FRED API Connection: Failed - {str(e)}")
        
        # Performance Metrics
        if data_fetcher.fetch_log:
            st.subheader("⚡ Performance Metrics")
            fetch_df = pd.DataFrame(data_fetcher.fetch_log)
            
            if 'response_time' in fetch_df.columns:
                avg_response_time = fetch_df['response_time'].mean()
                st.metric("Average API Response Time", f"{avg_response_time:.2f}s")
                
                # Response time distribution
                fig_hist = px.histogram(
                    fetch_df[fetch_df['response_time'].notna()], 
                    x='response_time',
                    title="API Response Time Distribution",
                    nbins=10
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Configuration Info
        st.subheader("🔧 Configuration")
        st.write(f"**FRED API Key:** {'✅ Configured' if fred_api_key else '❌ Missing'}")
        st.write(f"**Total Indicators:** {len(indicators)}")
        st.write(f"**Selected Time Horizon:** {selected_horizon}")
        st.write(f"**Debug Mode:** {'On' if debug_mode else 'Off'}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source:** Federal Reserve Economic Data (FRED)")

if __name__ == "__main__":
    main()
