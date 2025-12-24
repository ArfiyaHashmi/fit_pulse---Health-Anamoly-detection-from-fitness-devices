import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from prophet import Prophet
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# ============================================================================
# ANOMALY DETECTION METHODS
# ============================================================================

class ThresholdAnomalyDetector:
    """Rule-based anomaly detection using configurable thresholds."""
    
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'Heart rate outside normal resting range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,
                'sustained_minutes': 5,
                'description': 'Unrealistic step count detected'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,
                'max_threshold': 720,
                'sustained_minutes': 0,
                'description': 'Unusual sleep duration'
            }
        }
        self.detected_anomalies = []
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect threshold-based anomalies in the data."""
        
        report = {
            'method': 'Threshold-Based',
            'data_type': data_type,
            'total_records': len(df),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'threshold_info': {}
        }
        
        if data_type not in self.threshold_rules:
            return df, report
        
        rule = self.threshold_rules[data_type]
        metric_col = rule['metric_name']
        
        if metric_col not in df.columns:
            return df, report
        
        df_result = df.copy()
        df_result['threshold_anomaly'] = False
        df_result['anomaly_reason'] = ''
        df_result['severity'] = 'Normal'
        
        too_high = df_result[metric_col] > rule['max_threshold']
        too_low = df_result[metric_col] < rule['min_threshold']
        
        if rule['sustained_minutes'] > 0:
            window_size = rule['sustained_minutes']
            too_high_sustained = too_high.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            too_low_sustained = too_low.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            
            df_result.loc[too_high_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_high_sustained, 'anomaly_reason'] = f'High {metric_col}'
            df_result.loc[too_high_sustained, 'severity'] = 'High'
            
            df_result.loc[too_low_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_low_sustained, 'anomaly_reason'] = f'Low {metric_col}'
            df_result.loc[too_low_sustained, 'severity'] = 'Medium'
        else:
            df_result.loc[too_high, 'threshold_anomaly'] = True
            df_result.loc[too_high, 'anomaly_reason'] = f'Excessive {metric_col}'
            df_result.loc[too_high, 'severity'] = 'Medium'
            
            df_result.loc[too_low, 'threshold_anomaly'] = True
            df_result.loc[too_low, 'anomaly_reason'] = f'Insufficient {metric_col}'
            df_result.loc[too_low, 'severity'] = 'High'
        
        anomaly_count = df_result['threshold_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['threshold_info'] = {
            'min_threshold': rule['min_threshold'],
            'max_threshold': rule['max_threshold'],
            'sustained_minutes': rule['sustained_minutes']
        }
        
        return df_result, report


class ResidualAnomalyDetector:
    """Model-based anomaly detection using Prophet forecast residuals."""
    
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.prophet_models = {}
        self.detected_anomalies = []
    
    def fit_prophet_model(self, df: pd.DataFrame, data_type: str, 
                         forecast_periods: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """Fit Prophet model and generate predictions."""
        
        report = {
            'method': 'Prophet Residual-Based',
            'data_type': data_type,
            'success': False,
            'mae': 0,
            'rmse': 0
        }
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        if data_type not in metric_columns:
            return pd.DataFrame(), report
        
        metric_col = metric_columns[data_type]
        
        if metric_col not in df.columns:
            return pd.DataFrame(), report
        
        try:
            # Prepare Prophet data
            prophet_df = pd.DataFrame({
                'ds': df['timestamp'],
                'y': df[metric_col]
            }).dropna()
            
            if len(prophet_df) < 2:
                return pd.DataFrame(), report
            
            # Fit model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            with st.spinner(f"Training Prophet model for {data_type}..."):
                model.fit(prophet_df)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=forecast_periods, freq='min')
            forecast = model.predict(future)
            
            # Calculate residuals
            merged = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                     on='ds', how='left')
            merged['residual'] = merged['y'] - merged['yhat']
            merged['residual_abs'] = np.abs(merged['residual'])
            
            self.prophet_models[data_type] = {
                'model': model,
                'forecast': forecast,
                'residuals': merged
            }
            
            mae = merged['residual_abs'].mean()
            rmse = np.sqrt((merged['residual'] ** 2).mean())
            
            report.update({
                'success': True,
                'mae': float(mae),
                'rmse': float(rmse)
            })
            
            return forecast, report
            
        except Exception as e:
            report['error'] = str(e)
            return pd.DataFrame(), report
    
    def detect_anomalies_from_prophet(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies by comparing actual vs predicted values."""
        
        report = {
            'method': 'Prophet Residual-Based',
            'data_type': data_type,
            'threshold_std': self.threshold_std,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        if data_type not in self.prophet_models:
            return df, report
        
        model_data = self.prophet_models[data_type]
        forecast_df = model_data['forecast']
        residuals_df = model_data['residuals']
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        metric_col = metric_columns.get(data_type)
        if not metric_col:
            return df, report
        
        df_result = df.copy()
        df_result = df_result.sort_values('timestamp').reset_index(drop=True)
        
        # Merge with forecast
        forecast_aligned = forecast_df.copy()
        forecast_aligned = forecast_aligned.rename(columns={'ds': 'timestamp', 'yhat': 'predicted'})
        
        df_result = df_result.merge(
            forecast_aligned[['timestamp', 'predicted', 'yhat_lower', 'yhat_upper']], 
            on='timestamp', 
            how='left'
        )
        
        # Calculate residuals
        df_result['residual'] = df_result[metric_col] - df_result['predicted']
        
        residual_mean = df_result['residual'].mean()
        residual_std = df_result['residual'].std()
        
        threshold = self.threshold_std * residual_std
        df_result['residual_anomaly'] = np.abs(df_result['residual']) > threshold
        
        # Check if outside confidence interval
        outside_interval = (df_result[metric_col] > df_result['yhat_upper']) | \
                          (df_result[metric_col] < df_result['yhat_lower'])
        
        df_result['residual_anomaly'] = df_result['residual_anomaly'] | outside_interval
        df_result['residual_anomaly_reason'] = ''
        df_result.loc[df_result['residual_anomaly'], 'residual_anomaly_reason'] = 'Deviates from predicted trend'
        
        # Assign severity based on residual magnitude
        df_result['severity'] = 'Normal'
        df_result.loc[df_result['residual_anomaly'] & (np.abs(df_result['residual']) > 2*threshold), 'severity'] = 'High'
        df_result.loc[df_result['residual_anomaly'] & (np.abs(df_result['residual']) <= 2*threshold), 'severity'] = 'Medium'
        
        anomaly_count = df_result['residual_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['residual_stats'] = {
            'mean': float(residual_mean),
            'std': float(residual_std),
            'threshold': float(threshold)
        }
        
        return df_result, report


class ClusterAnomalyDetector:
    """Cluster-based anomaly detection."""
    
    def __init__(self, outlier_threshold: float = 0.05):
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        self.cluster_model = None
        self.cluster_labels = None
    
    def extract_features(self, df: pd.DataFrame, data_type: str, 
                        window_size: int = 30) -> pd.DataFrame:
        """Extract time-series features from data."""
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        metric_col = metric_columns.get(data_type)
        if not metric_col or metric_col not in df.columns:
            return pd.DataFrame()
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        features_list = []
        
        # Create sliding windows and extract features
        for i in range(0, len(df_sorted) - window_size + 1, window_size // 2):
            window = df_sorted.iloc[i:i+window_size][metric_col].values
            
            if len(window) < window_size:
                continue
            
            features = {
                'window_id': i // (window_size // 2),
                'mean': np.mean(window),
                'std': np.std(window),
                'median': np.median(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window),
                'skewness': stats.skew(window),
                'kurtosis': stats.kurtosis(window),
                'entropy': -np.sum(np.histogram(window, bins=10)[0]/len(window) * 
                          np.log(np.histogram(window, bins=10)[0]/len(window) + 1e-10))
            }
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def perform_clustering(self, feature_matrix: pd.DataFrame, 
                          method: str = 'kmeans', n_clusters: int = 3) -> np.ndarray:
        """Perform clustering on feature matrix."""
        
        if feature_matrix.empty:
            return np.array([])
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        if method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # dbscan
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
        
        self.cluster_labels = self.cluster_model.fit_predict(features_scaled)
        return self.cluster_labels
    
    def detect_cluster_outliers(self, feature_matrix: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies based on cluster membership."""
        
        report = {
            'method': 'Cluster-Based',
            'data_type': data_type,
            'total_clusters': 0,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        if len(cluster_labels) == 0:
            return feature_matrix, report
        
        df_result = feature_matrix.copy()
        df_result['cluster'] = cluster_labels
        
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        total_points = len(cluster_labels)
        
        anomalous_clusters = []
        
        # Find small clusters and noise points
        for cluster_id, size in cluster_sizes.items():
            cluster_percentage = size / total_points
            
            if cluster_id == -1:  # DBSCAN noise points
                anomalous_clusters.append(cluster_id)
            elif cluster_percentage < self.outlier_threshold:
                anomalous_clusters.append(cluster_id)
        
        # Mark anomalies
        df_result['cluster_anomaly'] = df_result['cluster'].isin(anomalous_clusters)
        df_result['cluster_anomaly_reason'] = ''
        df_result['severity'] = 'Normal'
        
        for cluster_id in anomalous_clusters:
            if cluster_id == -1:
                reason = 'Noise point (DBSCAN)'
            else:
                reason = f'Belongs to small cluster #{cluster_id}'
            
            mask = df_result['cluster'] == cluster_id
            df_result.loc[mask, 'cluster_anomaly_reason'] = reason
            df_result.loc[mask, 'severity'] = 'High'
        
        anomaly_count = df_result['cluster_anomaly'].sum()
        report['total_clusters'] = int(len(cluster_sizes))
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['cluster_distribution'] = {str(k): int(v) for k, v in cluster_sizes.items()}
        report['anomalous_clusters'] = [int(c) for c in anomalous_clusters]
        
        return df_result, report


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def create_sample_data_with_anomalies() -> Dict[str, pd.DataFrame]:
    """Create sample health data with intentional anomalies."""
    
    timestamps = pd.date_range(start='2024-01-15 08:00:00', 
                               end='2024-01-15 20:00:00', freq='1min')
    
    base_hr = 70
    hr_data = []
    
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        hr = base_hr
        
        if 9 <= time_of_day < 10:
            hr = 110 + np.random.normal(0, 5)
        elif 14 <= time_of_day < 15:
            hr = 95 + np.random.normal(0, 5)
        else:
            hr = 70 + np.random.normal(0, 3)
        
        # Anomalies
        if 11.5 <= time_of_day < 12:
            hr = 135 + np.random.normal(0, 5)
        if 16 <= time_of_day < 16.3:
            hr = 35 + np.random.normal(0, 2)
        if 18.5 <= time_of_day < 18.6:
            hr = 150
        
        hr_data.append(max(30, min(220, hr)))
    
    heart_rate_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': hr_data
    })
    
    step_timestamps = pd.date_range(start='2024-01-15 08:00:00',
                                   end='2024-01-15 20:00:00', freq='5min')
    
    step_data = []
    for i, ts in enumerate(step_timestamps):
        time_of_day = ts.hour + ts.minute / 60
        
        if 8 <= time_of_day < 9:
            steps = 50 + np.random.randint(-10, 10)
        elif 12 <= time_of_day < 13:
            steps = 80 + np.random.randint(-15, 15)
        elif 17 <= time_of_day < 18:
            steps = 100 + np.random.randint(-20, 20)
        else:
            steps = 20 + np.random.randint(-5, 5)
        
        # Anomaly
        if 15 <= time_of_day < 15.2:
            steps = 1200
        
        step_data.append(max(0, steps))
    
    steps_df = pd.DataFrame({
        'timestamp': step_timestamps,
        'step_count': step_data
    })
    
    return {
        'heart_rate': heart_rate_df,
        'steps': steps_df
    }


def main():
    st.set_page_config(
        page_title="Milestone 3 - Anomaly Detection",
        page_icon="🚨",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stMetric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🚨 Milestone 3: Anomaly Detection (All 3 Methods)")
    st.markdown("**Threshold-Based • Residual-Based • Cluster-Based Detection**")
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        detection_method = st.selectbox(
            "Select Detection Method",
            ["Threshold-Based", "Residual-Based (Prophet)", "Cluster-Based"]
        )
        
        st.divider()
        
        if detection_method == "Threshold-Based":
            st.subheader("Threshold Settings")
            hr_min = st.slider("Min Heart Rate (bpm)", 20, 60, 40)
            hr_max = st.slider("Max Heart Rate (bpm)", 100, 180, 120)
            sustained = st.slider("Sustained Duration (min)", 0, 30, 10)
        
        elif detection_method == "Residual-Based (Prophet)":
            st.subheader("Prophet Settings")
            threshold_std = st.slider("Threshold (σ)", 1.0, 5.0, 3.0, 0.5)
            forecast_periods = st.slider("Forecast Periods", 20, 100, 50)
        
        else:  # Cluster-Based
            st.subheader("Cluster Settings")
            clustering_method = st.selectbox("Clustering Method", ["KMeans", "DBSCAN"])
            if clustering_method == "KMeans":
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            outlier_threshold = st.slider("Outlier Threshold (%)", 1, 20, 5)
        
        st.divider()
        st.info("💡 Adjust parameters to customize anomaly detection")
    
    # Run analysis
    if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            data = create_sample_data_with_anomalies()
            
            # ========== THRESHOLD-BASED DETECTION ==========
            if detection_method == "Threshold-Based":
                st.subheader("📊 Threshold-Based Detection Results")
                
                threshold_detector = ThresholdAnomalyDetector()
                threshold_detector.threshold_rules['heart_rate']['min_threshold'] = hr_min
                threshold_detector.threshold_rules['heart_rate']['max_threshold'] = hr_max
                threshold_detector.threshold_rules['heart_rate']['sustained_minutes'] = sustained
                
                df_heart, report_heart = threshold_detector.detect_anomalies(
                    data['heart_rate'], 'heart_rate'
                )
                df_steps, report_steps = threshold_detector.detect_anomalies(
                    data['steps'], 'steps'
                )
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                total_anomalies = report_heart['anomalies_detected'] + report_steps['anomalies_detected']
                col1.metric("Total Anomalies", total_anomalies)
                col2.metric("Heart Rate Issues", report_heart['anomalies_detected'])
                col3.metric("Step Issues", report_steps['anomalies_detected'])
                col4.metric("Detection Rate", f"{(total_anomalies / (len(df_heart) + len(df_steps)) * 100):.2f}%")
                
                st.markdown("---")
                
                # Heart Rate Visualization
                st.subheader("❤️ Heart Rate Anomalies")
                fig_hr = go.Figure()
                
                normal_hr = df_heart[~df_heart['threshold_anomaly']]
                anomaly_hr = df_heart[df_heart['threshold_anomaly']]
                
                fig_hr.add_trace(go.Scatter(
                    x=normal_hr['timestamp'], y=normal_hr['heart_rate'],
                    mode='lines', name='Normal', line=dict(color='#3498db', width=2)
                ))
                
                if len(anomaly_hr) > 0:
                    fig_hr.add_trace(go.Scatter(
                        x=anomaly_hr['timestamp'], y=anomaly_hr['heart_rate'],
                        mode='markers', name='Anomalies',
                        marker=dict(color='#e74c3c', size=12, symbol='x')
                    ))
                
                fig_hr.add_hline(y=hr_max, line_dash="dash", line_color="red", 
                                annotation_text=f"Max ({hr_max})")
                fig_hr.add_hline(y=hr_min, line_dash="dash", line_color="red", 
                                annotation_text=f"Min ({hr_min})")
                
                fig_hr.update_layout(
                    title="Heart Rate with Threshold Anomalies",
                    xaxis_title="Time", yaxis_title="BPM",
                    height=400, hovermode='x unified', template='plotly_white'
                )
                st.plotly_chart(fig_hr, use_container_width=True)
                
                st.markdown("---")
                
                # Anomaly Details
                st.subheader("📋 Anomaly Details")
                if len(anomaly_hr) > 0:
                    display_df = anomaly_hr[['timestamp', 'heart_rate', 'anomaly_reason', 'severity']].copy()
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No anomalies detected with current thresholds")
            
            # ========== RESIDUAL-BASED DETECTION ==========
            elif detection_method == "Residual-Based (Prophet)":
                st.subheader("📈 Residual-Based Detection (Prophet)")
                
                residual_detector = ResidualAnomalyDetector(threshold_std=threshold_std)
                
                # Fit Prophet model
                forecast, prophet_report = residual_detector.fit_prophet_model(
                    data['heart_rate'], 'heart_rate', forecast_periods
                )
                
                if prophet_report['success']:
                    # Detect anomalies
                    df_residual, residual_report = residual_detector.detect_anomalies_from_prophet(
                        data['heart_rate'], 'heart_rate'
                    )
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Anomalies Detected", residual_report['anomalies_detected'])
                    col2.metric("MAE", f"{prophet_report['mae']:.2f} bpm")
                    col3.metric("RMSE", f"{prophet_report['rmse']:.2f} bpm")
                    col4.metric("Threshold (σ)", threshold_std)
                    
                    st.markdown("---")
                    
                    # Visualization
                    st.subheader("❤️ Heart Rate with Prophet Forecast")
                    fig = go.Figure()
                    
                    # Actual data
                    fig.add_trace(go.Scatter(
                        x=df_residual['timestamp'], y=df_residual['heart_rate'],
                        mode='markers', name='Actual',
                        marker=dict(size=6, color='#3498db')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', name='Forecast',
                        line=dict(color='#2ecc71', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'],
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'],
                        mode='lines', fill='tonexty',
                        fillcolor='rgba(46, 204, 113, 0.2)',
                        line=dict(width=0), name='95% Confidence'
                    ))
                    
                    # Mark anomalies
                    anomalies_residual = df_residual[df_residual['residual_anomaly']]
                    if len(anomalies_residual) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomalies_residual['timestamp'], y=anomalies_residual['heart_rate'],
                            mode='markers', name='Anomalies',
                            marker=dict(color='#e74c3c', size=12, symbol='diamond')
                        ))
                    
                    fig.update_layout(
                        title="Prophet Forecast with Residual Anomalies",
                        xaxis_title="Time", yaxis_title="BPM",
                        height=400, hovermode='x unified', template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Residual Statistics
                    st.subheader("📊 Residual Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Residual", f"{residual_report['residual_stats']['mean']:.3f}")
                    col2.metric("Std Residual", f"{residual_report['residual_stats']['std']:.3f}")
                    col3.metric("Threshold", f"{residual_report['residual_stats']['threshold']:.3f}")
                    
                    st.markdown("---")
                    
                    # Anomaly Details
                    st.subheader("📋 Anomaly Details")
                    if len(anomalies_residual) > 0:
                        display_df = anomalies_residual[['timestamp', 'heart_rate', 'predicted', 'residual', 'severity']].copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No anomalies detected with current threshold")
                else:
                    st.error("❌ Prophet model training failed.")
                    if 'error' in prophet_report:
                        st.warning(f"Error details: {prophet_report['error']}")
                    st.info("Try adjusting forecast periods or threshold settings.")
                st.markdown("---")
        st.success("✅ Analysis Complete!")
        st.balloons()
if __name__ == "__main__":
    main()
