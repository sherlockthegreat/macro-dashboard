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
import io

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
        self.fetch_log = []
        
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
        self.signal_log = []
    
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
                    'signal': signals[period].value[1],
                    'signal_score': signals[period].value[0]
                }
                        
            except Exception as e:
                signals[period] = SignalColor.YELLOW
                signal_details[period] = {'error': str(e)}
        
        self.signal_log.append({
            'indicator': indicator_name,
            'status': 'Success',
            'data_points': len(data),
            'latest_date': data['date'].max(),
            'signal_details': signal_details
        })
                
        return signals

def get_sample_indicators():
    return [
        IndicatorConfig("Federal Funds Rate", "Policy", "FEDFUNDS", "lower_is_better"),
        IndicatorConfig("Unemployment Rate", "Economy", "UNRATE", "lower_is_better"),
        IndicatorConfig("Consumer Price Index", "Economy", "CPIAUCSL", "lower_is_better"),
        IndicatorConfig("Industrial Production", "Economy", "INDPRO", "higher_is_better"),
        IndicatorConfig("Consumer Sentiment", "Economy", "UMCSENT", "higher_is_better"),
        IndicatorConfig("10-Year Treasury Rate", "Financial Markets", "GS10", "lower_is_better"),
        IndicatorConfig("VIX Volatility Index", "Market Structure", "VIXCLS", "lower_is_better"),
        IndicatorConfig("Real GDP", "Economy", "GDPC1", "higher_is_better"),
        IndicatorConfig("S&P 500", "Market Structure", "SP500", "higher_is_better"),
        IndicatorConfig("Dollar Index", "Currency", "DTWEXBGS", "higher_is_better"),
    ]

def create_comprehensive_csv(dashboard_data, signal_generator, data_fetcher, raw_data_store, categories):
    """Create comprehensive CSV with all metrics"""
    
    # 1. Main Dashboard Metrics
    main_df = pd.DataFrame(dashboard_data)
    
    # 2. Detailed Signal Analysis
    detailed_signals = []
    for log_entry in signal_generator.signal_log:
        if 'signal_details' in log_entry:
            for period, details in log_entry['signal_details'].items():
                if 'error' not in details:
                    detailed_signals.append({
                        'Indicator': log_entry['indicator'],
                        'Time_Horizon': period,
                        'Current_Value': details.get('current_value', 'N/A'),
                        'Historical_Value': details.get('historical_value', 'N/A'),
                        'Percent_Change': details.get('pct_change', 'N/A'),
                        'Signal': details.get('signal', 'N/A'),
                        'Signal_Score': details.get('signal_score', 'N/A'),
                        'Data_Points': log_entry['data_points'],
                        'Latest_Date': log_entry.get('latest_date', 'N/A')
                    })
    
    detailed_df = pd.DataFrame(detailed_signals)
    
    # 3. API Performance Metrics
    api_df = pd.DataFrame(data_fetcher.fetch_log)
    
    # 4. Category Summary
    category_summary = []
    for category, scores in categories.items():
        avg_score = np.mean(scores)
        if avg_score > 0.5:
            signal = "Good"
        elif avg_score > 0:
            signal = "Uncertainty"
        elif avg_score > -0.5:
            signal = "Caution"
        else:
            signal = "Bad"
            
        category_summary.append({
            'Category': category,
            'Average_Score': round(avg_score, 3),
            'Signal': signal,
            'Indicator_Count': len(scores)
        })
    
    category_df = pd.DataFrame(category_summary)
    
    # 5. Data Quality Summary
    total_indicators = len(dashboard_data)
    indicators_with_data = len([d for d in dashboard_data if d['Data Points'] > 0])
    successful_api_calls = len([log for log in data_fetcher.fetch_log if log['status'] == 'Success'])
    total_api_calls = len(data_fetcher.fetch_log)
    
    quality_metrics = pd.DataFrame([{
        'Metric': 'Total Indicators',
        'Value': total_indicators
    }, {
        'Metric': 'Indicators with Data',
        'Value': indicators_with_data
    }, {
        'Metric': 'Data Coverage %',
        'Value': round((indicators_with_data / total_indicators * 100) if total_indicators > 0 else 0, 2)
    }, {
        'Metric': 'API Success Rate %',
        'Value': round((successful_api_calls / total_api_calls * 100) if total_api_calls > 0 else 0, 2)
    }, {
        'Metric': 'Total API Calls',
        'Value': total_api_calls
    }, {
        'Metric': 'Successful API Calls',
        'Value': successful_api_calls
    }])
    
    # 6. Raw Data Summary
    raw_data_summary = []
    for indicator_name, data in raw_data_store.items():
        if data is not None and len(data) > 0:
            raw_data_summary.append({
                'Indicator': indicator_name,
                'Total_Data_Points': len(data),
                'First_Date': data['date'].min(),
                'Last_Date': data['date'].max(),
                'Current_Value': data['value'].iloc[-1],
                'Min_Value': data['value'].min(),
                'Max_Value': data['value'].max(),
                'Mean_Value': round(data['value'].mean(), 4),
                'Std_Dev': round(data['value'].std(), 4)
            })
    
    raw_summary_df = pd.DataFrame(raw_data_summary)
    
    return {
        'Dashboard_Summary': main_df,
        'Detailed_Signals': detailed_df,
        'API_Performance': api_df,
        'Category_Summary': category_df,
        'Data_Quality_Metrics': quality_metrics,
        'Raw_Data_Statistics': raw_summary_df
    }

def main():
    st.title("📊 V-Macro Dashboard")
    st.markdown("**Real-time Economic Indicator Tracking & Signal Generation**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    fred_api_key = st.secrets.get("FRED_API_KEY", "")
    
    if not fred_api_key:
        st.error("🔑 FRED API Key not configured. Please add it to your Streamlit secrets.")
        st.info("Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Initialize components
    data_fetcher = DataFetcher(fred_api_key)
    signal_generator = SignalGenerator()
    indicators = get_sample_indicators()
    
    # Sidebar options
    time_horizons = ['< 3 months', '3-6 months', '6-12 months', '12-24 months']
    selected_horizon = st.sidebar.selectbox("📅 Select Time Horizon", time_horizons)
    
    # CSV Export Options
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export Options")
    export_type = st.sidebar.selectbox(
        "Select Export Type:",
        ["Complete Dashboard Metrics", "Detailed Signals Only", "Raw Data Only", "API Performance Only"]
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Backend Data", "📈 Raw Data", "⚙️ System Health", "📥 Export Data"])
    
    # Data processing
    dashboard_data = []
    categories = {}
    raw_data_store = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, indicator in enumerate(indicators):
        progress_bar.progress((i + 1) / len(indicators))
        status_text.text(f"Processing: {indicator.name}")
        
        data = data_fetcher.fetch_fred_data(indicator.fred_series_id)
        raw_data_store[indicator.name] = data
        
        signals = signal_generator.calculate_signals(data, indicator.signal_logic, indicator.name)
        current_signal = signals.get(selected_horizon, SignalColor.YELLOW)
        
        if indicator.category not in categories:
            categories[indicator.category] = []
        categories[indicator.category].append(current_signal.value[0])
        
        dashboard_data.append({
            'Indicator': indicator.name,
            'Category': indicator.category,
            'FRED_Series_ID': indicator.fred_series_id,
            'Signal_Logic': indicator.signal_logic,
            'Signal': current_signal.value[1],
            'Signal_Score': current_signal.value[0],
            'Color': current_signal.value[2],
            'Data Points': len(data) if data is not None else 0,
            'Latest Date': data['date'].max().strftime('%Y-%m-%d') if data is not None and len(data) > 0 else 'No Data',
            'Latest_Value': data['value'].iloc[-1] if data is not None and len(data) > 0 else 'No Data'
        })
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(dashboard_data)
    
    with tab1:
        # Main dashboard (keep existing code)
        st.header(f"Dashboard - {selected_horizon}")
        
        # Category summary
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
        
        st.subheader("📈 Category Overview")
        if len(category_df) > 0:
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
        
        st.subheader("📊 Detailed Indicators")
        def color_signals(val):
            if val == 'Good':
                return 'background-color: #90EE90'
            elif val == 'Bad':
                return 'background-color: #FFB6C1'
            elif val == 'Caution':
                return 'background-color: #FFE4B5'
            else:
                return 'background-color: #FFFFE0'
        
        styled_df = df[['Indicator', 'Category', 'Signal', 'Data Points', 'Latest Date', 'Latest_Value']].style.applymap(
            color_signals, subset=['Signal']
        )
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        # Backend data (keep existing)
        st.header("🔍 Backend Data & Signal Analysis")
        
        if data_fetcher.fetch_log:
            fetch_df = pd.DataFrame(data_fetcher.fetch_log)
            st.dataframe(fetch_df, use_container_width=True)
            
            success_rate = len(fetch_df[fetch_df['status'] == 'Success']) / len(fetch_df) * 100
            st.metric("API Success Rate", f"{success_rate:.1f}%")
        
        if signal_generator.signal_log:
            for log_entry in signal_generator.signal_log:
                with st.expander(f"📊 {log_entry['indicator']} - {log_entry['status']}"):
                    st.write(f"**Data Points:** {log_entry['data_points']}")
                    if 'latest_date' in log_entry:
                        st.write(f"**Latest Date:** {log_entry['latest_date']}")
                    
                    if 'signal_details' in log_entry:
                        details_df = pd.DataFrame.from_dict(log_entry['signal_details'], orient='index')
                        st.dataframe(details_df)
    
    with tab3:
        # Raw data viewer (keep existing)
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
            
            st.dataframe(raw_data.tail(50), use_container_width=True)
            
            fig = px.line(raw_data.tail(100), x='date', y='value', title=f"{selected_indicator} - Time Series")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # System health (keep existing)
        st.header("⚙️ System Health & Performance")
        
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
    
    with tab5:
        # CSV Export Tab
        st.header("📥 Export Dashboard Data")
        
        # Create comprehensive CSV data
        csv_data = create_comprehensive_csv(dashboard_data, signal_generator, data_fetcher, raw_data_store, categories)
        
        st.subheader("Available Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Complete Dashboard Export
            if export_type == "Complete Dashboard Metrics":
                st.write("**Complete Dashboard Metrics** - All data in multiple sheets")
                
                # Create Excel file with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, data in csv_data.items():
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="📥 Download Complete Dashboard (Excel)",
                    data=output.getvalue(),
                    file_name=f"macro_dashboard_complete_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Also provide CSV of main dashboard
                main_csv = csv_data['Dashboard_Summary'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Main Dashboard (CSV)",
                    data=main_csv,
                    file_name=f"macro_dashboard_main_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif export_type == "Detailed Signals Only":
                detailed_csv = csv_data['Detailed_Signals'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed Signals (CSV)",
                    data=detailed_csv,
                    file_name=f"detailed_signals_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif export_type == "Raw Data Only":
                # Combine all raw data
                all_raw_data = []
                for indicator_name, data in raw_data_store.items():
                    if data is not None:
                        data_copy = data.copy()
                        data_copy['Indicator'] = indicator_name
                        all_raw_data.append(data_copy)
                
                if all_raw_data:
                    combined_raw = pd.concat(all_raw_data, ignore_index=True)
                    raw_csv = combined_raw.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Raw Data (CSV)",
                        data=raw_csv,
                        file_name=f"all_raw_data_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            elif export_type == "API Performance Only":
                api_csv = csv_data['API_Performance'].to_csv(index=False)
                st.download_button(
                    label="📥 Download API Performance (CSV)",
                    data=api_csv,
                    file_name=f"api_performance_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Preview of selected export
            st.write("**Preview of Selected Export:**")
            
            if export_type == "Complete Dashboard Metrics":
                st.dataframe(csv_data['Dashboard_Summary'].head(), use_container_width=True)
            elif export_type == "Detailed Signals Only":
                st.dataframe(csv_data['Detailed_Signals'].head(), use_container_width=True)
            elif export_type == "Raw Data Only":
                if raw_data_store:
                    first_indicator = list(raw_data_store.keys())[0]
                    if raw_data_store[first_indicator] is not None:
                        st.dataframe(raw_data_store[first_indicator].head(), use_container_width=True)
            elif export_type == "API Performance Only":
                st.dataframe(csv_data['API_Performance'].head(), use_container_width=True)
        
        # Data summary
        st.subheader("📊 Export Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Indicators", len(dashboard_data))
        
        with col2:
            st.metric("Data Points Available", sum([d['Data Points'] for d in dashboard_data]))
        
        with col3:
            st.metric("Categories", len(categories))
        
        with col4:
            st.metric("API Calls Made", len(data_fetcher.fetch_log))
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source:** Federal Reserve Economic Data (FRED)")

if __name__ == "__main__":
    main()
