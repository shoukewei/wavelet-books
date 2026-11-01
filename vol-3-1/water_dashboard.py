import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from water_preprocessor import WaterQualityPreprocessor
from water_denoiser import AdvancedWaveletDenoiser
from denoise_metrics import calculate_denoising_metrics

import warnings
warnings.filterwarnings('ignore')

class WaterQualityDashboard:
    def __init__(self):
        self.preprocessor = WaterQualityPreprocessor()
        self.denoiser = AdvancedWaveletDenoiser()
        
        # WHO/EPA water quality standards
        self.quality_standards = {
            'pH': {'min': 6.5, 'max': 8.5, 'unit': ''},
            'DO': {'min': 5.0, 'max': np.inf, 'unit': 'mg/L'},
            'COD': {'min': 0, 'max': 15.0, 'unit': 'mg/L'},
            'EC': {'min': 0, 'max': 1500, 'unit': 'µS/cm'},
            'NH': {'min': 0, 'max': 0.5, 'unit': 'mg/L'},
            'TUR': {'min': 0, 'max': 4.0, 'unit': 'NTU'},
            'WT': {'min': 0, 'max': 25, 'unit': '°C'}
        }
        
        # Data sampling limits to prevent memory issues
        self.MAX_PLOT_POINTS = 2000  # Maximum points for time series plots
        self.MAX_TABLE_ROWS = 500    # Maximum rows for data tables
        self.MAX_CORR_POINTS = 1000  # Maximum points for correlation analysis
    
    def downsample_data(self, df, max_points=None):
        """Intelligently downsample data while preserving key features"""
        if max_points is None:
            max_points = self.MAX_PLOT_POINTS
            
        if len(df) <= max_points:
            return df
        
        # Calculate sampling ratio
        ratio = len(df) / max_points
        
        # Use systematic sampling to preserve temporal distribution
        indices = np.round(np.linspace(0, len(df) - 1, max_points)).astype(int)
        return df.iloc[indices]
    
    def load_and_process_data(self, uploaded_file):
        """Load and process uploaded data with memory optimization"""
        if uploaded_file is not None:
            try:
                # Read data in chunks to handle large files
                df_raw = pd.read_csv(uploaded_file)
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                df_raw = df_raw.set_index('timestamp')
                
                # Show data size warning
                data_size_mb = df_raw.memory_usage(deep=True).sum() / 1024 / 1024
                if data_size_mb > 50:
                    st.warning(f"⚠️ Large dataset detected ({data_size_mb:.1f} MB). "
                             f"Visualizations will be downsampled for performance.")
                
                # Preprocess data
                df_clean = self.preprocessor.preprocess(df_raw)
                
                # Apply denoising
                df_denoised = self.denoiser.denoise_dataframe(df_clean)
                
                return df_raw, df_clean, df_denoised
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                return None, None, None
        
        return None, None, None
    
    def create_comparison_plot(self, df_raw, df_denoised, parameter):
        """Create comparison plot for raw vs denoised data with downsampling"""
        
        # Downsample data for plotting
        df_raw_plot = self.downsample_data(df_raw, self.MAX_PLOT_POINTS)
        df_denoised_plot = self.downsample_data(df_denoised.loc[df_raw_plot.index], self.MAX_PLOT_POINTS)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Raw {parameter} (sampled: {len(df_raw_plot)} points)', 
                          f'Denoised {parameter} (sampled: {len(df_denoised_plot)} points)'),
            vertical_spacing=0.1
        )
        
        # Raw data
        fig.add_trace(
            go.Scatter(
                x=df_raw_plot.index,
                y=df_raw_plot[parameter],
                mode='lines',
                name='Raw Data',
                line=dict(color='lightblue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Denoised data
        fig.add_trace(
            go.Scatter(
                x=df_denoised_plot.index,
                y=df_denoised_plot[parameter],
                mode='lines',
                name='Denoised Data',
                line=dict(color='darkblue', width=2)
            ),
            row=2, col=1
        )
        
        # Add quality standards if available
        if parameter in self.quality_standards:
            standards = self.quality_standards[parameter]
            
            if standards['max'] != np.inf:
                fig.add_hline(
                    y=standards['max'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Max: {standards['max']} {standards['unit']}",
                    row=1, col=1
                )
                fig.add_hline(
                    y=standards['max'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Max: {standards['max']} {standards['unit']}",
                    row=2, col=1
                )
            
            if standards['min'] > 0:
                fig.add_hline(
                    y=standards['min'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Min: {standards['min']} {standards['unit']}",
                    row=1, col=1
                )
                fig.add_hline(
                    y=standards['min'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Min: {standards['min']} {standards['unit']}",
                    row=2, col=1
                )
        
        fig.update_layout(
            title=f'{parameter} - Raw vs Denoised Comparison',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text=f"{parameter} ({self.quality_standards.get(parameter, {}).get('unit', '')})")
        
        return fig
    
    def create_correlation_matrix(self, df):
        """Create correlation matrix heatmap with data sampling"""
        # Limit to prevent memory issues
        df_corr = self.downsample_data(df, self.MAX_CORR_POINTS)
        correlation_matrix = df_corr.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Parameter Correlation Matrix (based on {len(df_corr)} samples)',
            height=500
        )
        
        return fig
    
    def detect_anomalies(self, df, parameter):
        """Detect anomalies using multiple methods"""
        data = df[parameter].values
        
        # Z-score method
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        z_anomalies = z_scores > 3
        
        # IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        iqr_anomalies = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
        
        # Combined anomalies
        anomalies = z_anomalies | iqr_anomalies
        
        return anomalies, z_scores
    
    def create_multi_parameter_plot(self, df, available_params):
        """Create multi-parameter plot with optimized data handling"""
        # Downsample data
        df_plot = self.downsample_data(df, self.MAX_PLOT_POINTS)
        
        # Limit number of parameters to prevent overload
        max_params = min(len(available_params), 7)  # Limit to 7 parameters
        params_to_plot = available_params[:max_params]
        
        if len(available_params) > max_params:
            st.info(f"📊 Showing first {max_params} parameters. Use parameter selection above to view others.")
        
        fig = make_subplots(
            rows=len(params_to_plot), cols=1,
            subplot_titles=[f"{param} ({len(df_plot)} points sampled)" for param in params_to_plot],
            vertical_spacing=0.02,
            shared_xaxes=True
        )
        
        for i, param in enumerate(params_to_plot, 1):
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot[param],
                    mode='lines',
                    name=param,
                    showlegend=False,
                    line=dict(width=1.5)
                ),
                row=i, col=1
            )
            
            fig.update_yaxes(
                title_text=f"{param} ({self.quality_standards.get(param, {}).get('unit', '')})",
                row=i, col=1
            )
        
        fig.update_layout(
            title="Multi-Parameter Overview - Denoised Data",
            height=200 * len(params_to_plot),
            showlegend=False
        )
        fig.update_xaxes(title_text="Time", row=len(params_to_plot), col=1)
        
        return fig
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.set_page_config(
            page_title="Water Quality Monitoring Dashboard",
            page_icon="💧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🌊 Advanced Water Quality Monitoring System")
        st.markdown("*Real-time analysis with wavelet denoising and anomaly detection*")
        
        # Sidebar for controls
        st.sidebar.header("📊 Dashboard Controls")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Water Quality Data",
            type=['csv'],
            help="Upload CSV file with timestamp and water quality parameters"
        )
        
        # Data sampling controls
        st.sidebar.subheader("⚙️ Performance Settings")
        
        custom_plot_points = st.sidebar.slider(
            "Max Plot Points",
            min_value=500,
            max_value=5000,
            value=self.MAX_PLOT_POINTS,
            step=250,
            help="Reduce for better performance with large datasets"
        )
        self.MAX_PLOT_POINTS = custom_plot_points
        
        if uploaded_file is not None:
            # Process data
            with st.spinner("Processing data..."):
                df_raw, df_clean, df_denoised = self.load_and_process_data(uploaded_file)
            
            if df_raw is not None and hasattr(df_raw, 'columns'):
                available_params = [col for col in df_raw.columns if col != 'timestamp']
                
                st.success(f"✅ Data loaded successfully! {len(df_raw):,} records processed.")
                st.info(f"📊 Visualizations limited to {self.MAX_PLOT_POINTS:,} points for performance.")
                
                # Parameter selection
                selected_param = st.sidebar.selectbox(
                    "Select Parameter for Analysis",
                    available_params,
                    help="Choose which water quality parameter to analyze"
                )
                
                # Time range selection with smart defaults
                min_date = df_raw.index.min().date()
                max_date = df_raw.index.max().date()
                
                # Default to last 7 days or full range if shorter
                days_available = (max_date - min_date).days
                default_days = min(7, days_available)
                default_start = max_date - timedelta(days=default_days)
                
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=[default_start, max_date],
                    min_value=min_date,
                    max_value=max_date,
                    help=f"Total data spans {days_available} days"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
                    df_filtered = df_raw[mask]
                    df_denoised_filtered = df_denoised.loc[df_raw.index[mask]]
                    
                    if len(df_filtered) == 0:
                        st.warning("⚠️ No data available for selected date range.")
                        return
                else:
                    df_filtered = df_raw
                    df_denoised_filtered = df_denoised
                
                # Main content area
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Time series comparison
                    st.subheader(f"📈 {selected_param} Time Series Analysis")
                    
                    if len(df_filtered) > 0:
                        comparison_fig = self.create_comparison_plot(
                            df_filtered, df_denoised_filtered, selected_param
                        )
                        st.plotly_chart(comparison_fig, width='stretch')
                        
                        # Anomaly detection
                        st.subheader("🚨 Anomaly Detection")
                        
                        anomalies, z_scores = self.detect_anomalies(df_denoised_filtered, selected_param)
                        
                        if np.any(anomalies):
                            st.warning(f"⚠️ {np.sum(anomalies)} anomalies detected in {selected_param}!")
                            
                            # Show anomaly timestamps (limited)
                            anomaly_indices = np.where(anomalies)[0]
                            anomaly_times = df_denoised_filtered.index[anomalies]
                            anomaly_values = df_denoised_filtered[selected_param].iloc[anomalies]
                            
                            # Limit anomaly display
                            max_anomalies_display = min(100, len(anomaly_times))
                            
                            anomaly_df = pd.DataFrame({
                                'Timestamp': anomaly_times[:max_anomalies_display],
                                'Value': anomaly_values[:max_anomalies_display],
                                'Z-Score': z_scores[anomaly_indices][:max_anomalies_display]
                            })
                            
                            if len(anomaly_times) > max_anomalies_display:
                                st.info(f"Showing first {max_anomalies_display} of {len(anomaly_times)} anomalies")
                            
                            st.dataframe(anomaly_df, width='stretch')
                        else:
                            st.success("✅ No anomalies detected in the selected time range.")
                
                with col2:
                    # Statistics panel
                    st.subheader("📊 Statistics Summary")
                    
                    # Quality metrics
                    if selected_param in self.quality_standards:
                        standards = self.quality_standards[selected_param]
                        current_value = df_denoised_filtered[selected_param].iloc[-1]
                        
                        if standards['min'] <= current_value <= standards['max']:
                            st.success(f"✅ {selected_param}: {current_value:.2f} {standards['unit']} (NORMAL)")
                        else:
                            st.error(f"❌ {selected_param}: {current_value:.2f} {standards['unit']} (OUT OF RANGE)")
                    
                    # Basic statistics
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Latest'],
                        'Raw Data': [
                            f"{df_filtered[selected_param].mean():.2f}",
                            f"{df_filtered[selected_param].std():.2f}",
                            f"{df_filtered[selected_param].min():.2f}",
                            f"{df_filtered[selected_param].max():.2f}",
                            f"{df_filtered[selected_param].iloc[-1]:.2f}"
                        ],
                        'Denoised': [
                            f"{df_denoised_filtered[selected_param].mean():.2f}",
                            f"{df_denoised_filtered[selected_param].std():.2f}",
                            f"{df_denoised_filtered[selected_param].min():.2f}",
                            f"{df_denoised_filtered[selected_param].max():.2f}",
                            f"{df_denoised_filtered[selected_param].iloc[-1]:.2f}"
                        ]
                    })
                    
                    st.dataframe(stats_df, width='stretch')
                    
                    # Denoising performance metrics (on sampled data for performance)
                    sample_size = min(1000, len(df_filtered))
                    sample_indices = np.random.choice(len(df_filtered), sample_size, replace=False)
                    
                    metrics = calculate_denoising_metrics(
                        df_filtered[selected_param].iloc[sample_indices].values,
                        df_denoised_filtered[selected_param].iloc[sample_indices].values
                    )
                    
                    st.subheader("🔧 Denoising Performance")
                    st.metric("SNR Improvement", f"{metrics['snr_improvement_db']:.1f} dB")
                    st.metric("Signal Correlation", f"{metrics['correlation']:.3f}")
                    st.metric("Peak Preservation", f"{metrics['peak_preservation']:.1%}")
                    st.metric("Noise Reduction", f"{metrics['noise_reduction_ratio']:.3f}x")
                
                # Multi-parameter overview
                st.subheader("🌐 Multi-Parameter Overview")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 All Parameters", "🔗 Correlations", "📋 Data Table"])
                
                with tab1:
                    # Multi-parameter time series
                    fig_multi = self.create_multi_parameter_plot(df_denoised_filtered, available_params)
                    st.plotly_chart(fig_multi, width='stretch')
                
                with tab2:
                    # Correlation analysis
                    corr_fig = self.create_correlation_matrix(df_denoised_filtered)
                    st.plotly_chart(corr_fig, width='stretch')
                
                with tab3:
                    # Data table with filtering
                    st.subheader("📋 Recent Data Sample")
                    
                    # Show recent data sample
                    recent_data = self.downsample_data(
                        df_denoised_filtered.tail(min(2000, len(df_denoised_filtered))), 
                        self.MAX_TABLE_ROWS
                    ).reset_index()
                    
                    st.info(f"Showing {len(recent_data)} most recent samples")
                    
                    # Display data table
                    st.dataframe(recent_data, width='stretch', height=400)
                    
                    # Download option for full data
                    if st.button("📥 Generate Download Link for Full Dataset"):
                        csv = df_denoised_filtered.to_csv()
                        st.download_button(
                            label="Download Processed Data as CSV",
                            data=csv,
                            file_name=f"processed_water_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            else:
                st.error("❌ Error: Unable to process the uploaded file. Please check the format.")
                
        else:
            # Landing page
            st.info("👆 Please upload a water quality dataset to begin analysis.")
            
            # Sample data format
            st.subheader("📋 Expected Data Format")
            
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
                'pH': [7.2, 7.1, 7.3, 7.0, 7.4],
                'DO': [8.5, 8.2, 8.7, 8.1, 8.6],
                'COD': [12.3, 11.8, 12.1, 11.9, 12.5],
                'EC': [345, 352, 338, 349, 341],
                'NH': [0.23, 0.21, 0.25, 0.22, 0.24],
                'TUR': [2.1, 2.3, 1.9, 2.0, 2.2],
                'WT': [18.5, 18.7, 18.3, 18.6, 18.4]
            })
            
            st.dataframe(sample_data, width='stretch')
            
            # Performance tips
            st.subheader("⚡ Performance Tips")
            st.markdown("""
            - **Large datasets**: The dashboard automatically downsamples data for visualization
            - **Memory usage**: Reduce 'Max Plot Points' in the sidebar for better performance
            - **Time ranges**: Select shorter time periods for faster loading
            - **File size**: Consider splitting very large files (>100MB) into smaller chunks
            """)
            
            # System information
            st.subheader("ℹ️ System Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔧 Preprocessing**
                - Missing value interpolation
                - Outlier detection & removal
                - Consistent time resampling
                - Parameter-specific handling
                """)
            
            with col2:
                st.markdown("""
                **🌊 Wavelet Denoising**
                - Multi-scale decomposition
                - Adaptive thresholding
                - Parameter optimization
                - Signal preservation
                """)
            
            with col3:
                st.markdown("""
                **📊 Analysis & Alerts**
                - Real-time anomaly detection
                - Quality standard compliance
                - Multi-parameter correlation
                - Performance metrics
                """)

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = WaterQualityDashboard()
    dashboard.run_dashboard()