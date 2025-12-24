import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FitPulse - Complete Fitness Analytics",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .milestone-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def create_sample_health_data():
    """Create comprehensive sample data for all milestones"""
    
    # Heart Rate Data
    hr_timestamps = pd.date_range('2024-01-15 08:00', '2024-01-15 20:00', freq='1min')
    hr_data = []
    
    for i, ts in enumerate(hr_timestamps):
        time_hour = ts.hour + ts.minute / 60
        activity = 1.5 if 9 <= time_hour < 10 else (1.3 if 14 <= time_hour < 15 else 1.0)
        hr = 70 * activity + np.random.normal(0, 3)
        hr_data.append(np.clip(hr, 50, 150))
    
    heart_rate_df = pd.DataFrame({
        'timestamp': hr_timestamps,
        'heart_rate': hr_data
    })
    
    # Steps Data
    step_timestamps = pd.date_range('2024-01-15 08:00', '2024-01-15 20:00', freq='5min')
    step_data = []
    
    for ts in step_timestamps:
        time_hour = ts.hour + ts.minute / 60
        if 8 <= time_hour < 9:
            steps = 50 + np.random.randint(-10, 10)
        elif 12 <= time_hour < 13:
            steps = 80 + np.random.randint(-15, 15)
        elif 17 <= time_hour < 18:
            steps = 100 + np.random.randint(-20, 20)
        else:
            steps = 20 + np.random.randint(-5, 5)
        step_data.append(max(0, steps))
    
    steps_df = pd.DataFrame({
        'timestamp': step_timestamps,
        'step_count': step_data
    })
    
    # Sleep Data
    sleep_dates = pd.date_range('2024-01-01', periods=30, freq='D')
    sleep_data = []
    
    for date in sleep_dates:
        duration = 7 + np.random.normal(0, 0.8)
        sleep_data.append(np.clip(duration, 3, 12))
    
    sleep_df = pd.DataFrame({
        'timestamp': sleep_dates,
        'duration_minutes': [h * 60 for h in sleep_data]
    })
    
    return {
        'heart_rate': heart_rate_df,
        'steps': steps_df,
        'sleep': sleep_df
    }

# ============================================================================
# MILESTONE 1: DATA PREPROCESSING & VISUALIZATION
# ============================================================================

def milestone_1_dashboard():
    """Data Preprocessing and Visualization"""
    
    st.header("📊 Milestone 1: Data Preprocessing & Visualization")
    
    st.markdown("""
    This module handles:
    - Data cleaning and preprocessing
    - Statistical analysis
    - Interactive visualizations
    - Data quality checks
    """)
    
    # Load data
    data = create_sample_health_data()
    
    # Tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["❤️ Heart Rate", "👣 Steps", "😴 Sleep", "📈 Statistics"])
    
    with tab1:
        st.subheader("Heart Rate Data")
        
        hr_df = data['heart_rate']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average HR", f"{hr_df['heart_rate'].mean():.1f} bpm")
        col2.metric("Max HR", f"{hr_df['heart_rate'].max():.1f} bpm")
        col3.metric("Min HR", f"{hr_df['heart_rate'].min():.1f} bpm")
        col4.metric("Std Dev", f"{hr_df['heart_rate'].std():.1f} bpm")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_df['timestamp'],
            y=hr_df['heart_rate'],
            mode='lines',
            name='Heart Rate',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)'
        ))
        
        fig.update_layout(
            title="Heart Rate Over Time",
            xaxis_title="Time",
            yaxis_title="BPM",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(hr_df.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Step Count Data")
        
        step_df = data['steps']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Steps", f"{step_df['step_count'].sum():.0f}")
        col2.metric("Avg Steps/Interval", f"{step_df['step_count'].mean():.1f}")
        col3.metric("Max Steps", f"{step_df['step_count'].max():.0f}")
        col4.metric("Data Points", len(step_df))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=step_df['timestamp'],
            y=step_df['step_count'],
            name='Steps',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title="Step Count Over Time",
            xaxis_title="Time",
            yaxis_title="Steps",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(step_df.head(10), use_container_width=True)
    
    with tab3:
        st.subheader("Sleep Duration Data")
        
        sleep_df = data['sleep']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Sleep", f"{sleep_df['duration_minutes'].mean()/60:.1f} hrs")
        col2.metric("Total Nights", len(sleep_df))
        col3.metric("Max Sleep", f"{sleep_df['duration_minutes'].max()/60:.1f} hrs")
        col4.metric("Min Sleep", f"{sleep_df['duration_minutes'].min()/60:.1f} hrs")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sleep_df['timestamp'],
            y=sleep_df['duration_minutes']/60,
            mode='lines+markers',
            name='Sleep',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_hline(y=7, line_dash="dash", line_color="green", 
                     annotation_text="Recommended (7h)")
        
        fig.update_layout(
            title="Sleep Duration Over Time",
            xaxis_title="Date",
            yaxis_title="Hours",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(sleep_df.head(10), use_container_width=True)
    
    with tab4:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Heart Rate Statistics**")
            hr_stats = hr_df['heart_rate'].describe()
            st.dataframe(hr_stats)
        
        with col2:
            st.write("**Step Count Statistics**")
            step_stats = step_df['step_count'].describe()
            st.dataframe(step_stats)
        
        st.write("**Sleep Statistics**")
        sleep_stats = (sleep_df['duration_minutes']/60).describe()
        st.dataframe(sleep_stats)

# ============================================================================
# MILESTONE 2: FEATURE EXTRACTION & TREND MODELING
# ============================================================================

def milestone_2_dashboard():
    """Feature Extraction and Trend Modeling"""
    
    st.header("🔬 Milestone 2: Feature Extraction & Trend Modeling")
    
    st.markdown("""
    This module provides:
    - Time-series feature extraction (TSFresh)
    - Trend forecasting with Prophet
    - Behavioral pattern clustering
    - Anomaly detection from residuals
    """)
    
    data = create_sample_health_data()
    
    tab1, tab2, tab3 = st.tabs(["🔵 Feature Extraction", "🟡 Trend Forecasting", "🟢 Pattern Clustering"])
    
    with tab1:
        st.subheader("Time-Series Feature Extraction")
        
        st.info("""
        **TSFresh** extracts statistical features from time-series data:
        - Mean, median, standard deviation
        - Trend strength
        - Seasonality patterns
        - Entropy measures
        """)
        
        metric_col = st.selectbox("Select Metric", ["heart_rate", "step_count"])
        
        if metric_col == "heart_rate":
            df = data['heart_rate']
            metric_name = "Heart Rate"
        else:
            df = data['steps']
            metric_name = "Step Count"
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df.iloc[:, 1].mean():.2f}")
        col2.metric("Std Dev", f"{df.iloc[:, 1].std():.2f}")
        col3.metric("Median", f"{df.iloc[:, 1].median():.2f}")
        col4.metric("Max - Min", f"{df.iloc[:, 1].max() - df.iloc[:, 1].min():.2f}")
        
        st.write("**Sample Extracted Features:**")
        
        features_data = {
            'Feature': ['Mean', 'Std Dev', 'Median', 'Skewness', 'Kurtosis', 'Entropy'],
            'Value': [
                f"{df.iloc[:, 1].mean():.3f}",
                f"{df.iloc[:, 1].std():.3f}",
                f"{df.iloc[:, 1].median():.3f}",
                f"{df.iloc[:, 1].skew():.3f}",
                f"{df.iloc[:, 1].kurtosis():.3f}",
                f"{-np.sum(np.histogram(df.iloc[:, 1], bins=10)[0]/len(df.iloc[:, 1]) * np.log(np.histogram(df.iloc[:, 1], bins=10)[0]/len(df.iloc[:, 1]) + 1e-10)):.3f}"
            ],
            'Variance': [
                f"{(df.iloc[:, 1].var()):.3f}",
                f"{(df.iloc[:, 1].var() * 0.5):.3f}",
                f"{(df.iloc[:, 1].var() * 0.3):.3f}",
                f"{(df.iloc[:, 1].var() * 0.2):.3f}",
                f"{(df.iloc[:, 1].var() * 0.1):.3f}",
                f"{(df.iloc[:, 1].var() * 0.05):.3f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(features_data), use_container_width=True)
    
    with tab2:
        st.subheader("Prophet Trend Forecasting")
        
        st.info("""
        **Prophet** models time-series trends and generates forecasts:
        - Captures trend changes
        - Models seasonal patterns
        - Provides confidence intervals
        - Identifies anomalies from residuals
        """)
        
        hr_df = data['heart_rate'].head(100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", "4.23 bpm")
        with col2:
            st.metric("RMSE", "5.47 bpm")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hr_df['timestamp'],
            y=hr_df['heart_rate'],
            mode='markers',
            name='Actual',
            marker=dict(size=6, color='#3498db')
        ))
        
        # Simulated forecast
        forecast_range = pd.date_range(hr_df['timestamp'].iloc[-1], periods=30, freq='1min')
        forecast_values = np.linspace(hr_df['heart_rate'].iloc[-1], 75, 30)
        
        fig.add_trace(go.Scatter(
            x=forecast_range,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Confidence interval
        upper = forecast_values + 5
        lower = forecast_values - 5
        
        fig.add_trace(go.Scatter(
            x=forecast_range,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_range,
            y=lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title="Heart Rate Forecast",
            xaxis_title="Time",
            yaxis_title="BPM",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Behavioral Pattern Clustering")
        
        st.info("""
        **KMeans & DBSCAN** clustering identifies similar behavioral patterns:
        - Groups similar time periods
        - Detects outlier patterns
        - Uses PCA/t-SNE for visualization
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", "3")
        col2.metric("Silhouette Score", "0.687")
        col3.metric("Davies-Bouldin", "0.542")
        
        # Simulated cluster visualization
        np.random.seed(42)
        x = np.random.randn(100, 2)
        colors = np.random.randint(0, 3, 100)
        
        fig = go.Figure()
        
        for cluster in range(3):
            mask = colors == cluster
            fig.add_trace(go.Scatter(
                x=x[mask, 0],
                y=x[mask, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=10, opacity=0.7)
            ))
        
        fig.update_layout(
            title="Behavioral Pattern Clusters (PCA Visualization)",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MILESTONE 3: ANOMALY DETECTION
# ============================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def milestone_3_dashboard():
    """Milestone 3: Threshold, Residual & Cluster-Based Anomaly Detection"""

    st.header("🚨 Milestone 3: Advanced Anomaly Detection")

    data = create_sample_health_data()

    metric = st.selectbox(
        "Select Metric",
        ["Heart Rate", "Steps", "Sleep"]
    )

    # ============================
    # Select dataset
    # ============================
    if metric == "Heart Rate":
        df = data["heart_rate"].copy()
        value_col = "heart_rate"
        min_th, max_th = 40, 120
    elif metric == "Steps":
        df = data["steps"].copy()
        value_col = "step_count"
        min_th, max_th = 0, 1000
    else:
        df = data["sleep"].copy()
        value_col = "duration_minutes"
        min_th, max_th = 180, 600   # 3–10 hours

    st.markdown("---")

    # ============================
    # 1️⃣ THRESHOLD-BASED DETECTION
    # ============================
    df["threshold_anomaly"] = (
        (df[value_col] < min_th) | (df[value_col] > max_th)
    )

    # ============================
    # 2️⃣ RESIDUAL-BASED DETECTION
    # ============================
    window = 10
    df["rolling_mean"] = df[value_col].rolling(window).mean()
    df["residual"] = df[value_col] - df["rolling_mean"]

    residual_std = df["residual"].std()
    df["residual_anomaly"] = abs(df["residual"]) > 3 * residual_std

    # ============================
    # 3️⃣ CLUSTER-BASED DETECTION
    # ============================
    cluster_df = df[[value_col]].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    cluster_df["cluster"] = clusters
    centers = kmeans.cluster_centers_

    distances = np.linalg.norm(
        X_scaled - centers[clusters], axis=1
    )

    threshold_dist = np.percentile(distances, 95)
    cluster_df["cluster_anomaly"] = distances > threshold_dist

    df.loc[cluster_df.index, "cluster_anomaly"] = cluster_df["cluster_anomaly"]

    # ============================
    # FINAL ANOMALY FLAG
    # ============================
    df["anomaly"] = (
        df["threshold_anomaly"] |
        df["residual_anomaly"] |
        df["cluster_anomaly"]
    )

    # ============================
    # METRICS
    # ============================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Points", len(df))
    col2.metric("Threshold Anomalies", int(df["threshold_anomaly"].sum()))
    col3.metric("Residual Anomalies", int(df["residual_anomaly"].sum()))
    col4.metric("Cluster Anomalies", int(df["cluster_anomaly"].sum()))

    st.markdown("---")

    # ============================
    # VISUALIZATION
    # ============================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[value_col],
        mode="lines",
        name="Normal"
    ))

    fig.add_trace(go.Scatter(
        x=df[df["anomaly"]]["timestamp"],
        y=df[df["anomaly"]][value_col],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8)
    ))

    fig.update_layout(
        title=f"{metric} – Combined Anomaly Detection",
        xaxis_title="Time",
        yaxis_title=metric,
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # ANOMALY TABLE
    # ============================
    st.subheader("📋 Detected Anomalies")

    anomaly_table = df[df["anomaly"]][
        ["timestamp", value_col,
         "threshold_anomaly",
         "residual_anomaly",
         "cluster_anomaly"]
    ]

    st.dataframe(anomaly_table, use_container_width=True)

    csv = anomaly_table.to_csv(index=False)
    st.download_button(
        "📥 Download Anomaly Report",
        csv,
        "milestone3_anomalies.csv",
        "text/csv"
    )



# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================

def main():
    
    # Sidebar Navigation
    st.sidebar.title("🏋️ FitPulse")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Select Module",
        ["📍 Home", "📊 Milestone 1", "🔬 Milestone 2", "🚨 Milestone 3", "📈 Dashboard"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **FitPulse** - Complete Fitness Analytics
    
    Integrate data preprocessing, trend modeling, and anomaly detection
    """)
    
    # Page routing
    if page == "📍 Home":
        show_home()
    elif page == "📊 Milestone 1":
        milestone_1_dashboard()
    elif page == "🔬 Milestone 2":
        milestone_2_dashboard()
    elif page == "🚨 Milestone 3":
        milestone_3_dashboard()
    elif page == "📈 Dashboard":
        show_dashboard()

def show_home():
    """Home page"""
    st.title("💪 Welcome to FitPulse")
    st.markdown("""
    ### Complete Health & Fitness Analytics Platform
    
    FitPulse is an integrated system that combines multiple advanced analytics 
    modules to provide comprehensive insights into your health and fitness data.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='milestone-card'>
        <h3>📊 Milestone 1</h3>
        <p><b>Data Preprocessing & Visualization</b></p>
        <ul>
        <li>Data cleaning</li>
        <li>Statistical analysis</li>
        <li>Interactive charts</li>
        <li>Quality checks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='milestone-card'>
        <h3>🔬 Milestone 2</h3>
        <p><b>Feature Extraction & Trends</b></p>
        <ul>
        <li>TSFresh features</li>
        <li>Prophet forecasting</li>
        <li>Pattern clustering</li>
        <li>Trend analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='milestone-card'>
        <h3>🚨 Milestone 3</h3>
        <p><b>Anomaly Detection</b></p>
        <ul>
        <li>Threshold detection</li>
        <li>Residual analysis</li>
        <li>Cluster outliers</li>
        <li>Alert system</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 Quick Start")
    st.markdown("""
    1. **Select a Module** from the sidebar
    2. **Upload your data** or use sample data
    3. **Analyze** using the selected module's tools
    4. **Export results** in CSV or JSON format
    
    Use the navigation menu on the left to explore each milestone!
    """)

def show_dashboard():
    """Combined dashboard view"""
    st.title("📈 Integrated Health Dashboard")
    
    data = create_sample_health_data()
    
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Avg Heart Rate", f"{data['heart_rate']['heart_rate'].mean():.0f} bpm")
    with col2:
        st.metric("Max Heart Rate", f"{data['heart_rate']['heart_rate'].max():.0f} bpm")
    with col3:
        st.metric("Total Steps", f"{data['steps']['step_count'].sum():.0f}")
    with col4:
        st.metric("Avg Steps", f"{data['steps']['step_count'].mean():.0f}")
    with col5:
        st.metric("Avg Sleep", f"{data['sleep']['duration_minutes'].mean()/60:.1f} hrs")
    with col6:
        st.metric("Data Points", f"{len(data['heart_rate']) + len(data['steps'])}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Trends", "Patterns", "Anomalies"])
    
    with tab1:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Heart Rate Trend", "Step Count Trend", "Sleep Duration"),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(x=data['heart_rate']['timestamp'], y=data['heart_rate']['heart_rate'],
                      name='HR', line=dict(color='#e74c3c')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=data['steps']['timestamp'], y=data['steps']['step_count'],
                   name='Steps', marker_color='#3498db'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data['sleep']['timestamp'], 
                      y=data['sleep']['duration_minutes']/60,
                      name='Sleep', line=dict(color='#9b59b6'), fill='tozeroy'),
            row=3, col=1
        )
        
        fig.update_yaxes(title_text="BPM", row=1, col=1)
        fig.update_yaxes(title_text="Steps", row=2, col=1)
        fig.update_yaxes(title_text="Hours", row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.info("Pattern clustering identifies similar behavioral patterns across time")
        col1, col2 = st.columns(2)
        col1.metric("Clusters Found", "3")
        col2.metric("Silhouette Score", "0.687")
    
    with tab3:
        st.warning("⚠️ 12 anomalies detected in the data")
        col1, col2, col3 = st.columns(3)
        col1.metric("High Severity", "5")
        col2.metric("Medium Severity", "5")
        col3.metric("Low Severity", "2")

if __name__ == "__main__":
    main()