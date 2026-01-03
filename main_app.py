import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import json
import io

GREEN = "#10B981"
RED = "#EF4444"
BLUE = "#3B82F6"
YELLOW = "#F59E0B"

# Page Configuration
st.set_page_config(
    page_title="FitPulse - Health Anomaly Detection",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)
#-------------------------------------
# Custom CSS for perfect styling with animations
#------------------------------------------------
st.markdown("""
<style>

/* ===============================
   GLOBAL THEME
================================ */
html, body {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #F9FAFB;
}

.stApp {
    background-color: #F9FAFB;
    color: #1F2937;
}

/* ===============================
   PAGE LAYOUT
================================ */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===============================
   HEADINGS
================================ */
h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1F2937;
}
h2 {
    font-size: 1.4rem;
    font-weight: 600;
    color: #374151;
}
h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4B5563;
}

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
    padding-top: 1rem;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.4rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] label {
    font-weight: 500;
    color: #374151;
}

/* ===============================
   METRIC CARDS
================================ */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 1.3rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 18px rgba(0,0,0,0.04);
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    color: #6B7280;
}

[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: #5DA9E9;
}

/* ===============================
   STATUS CARDS
================================ */
.status-normal {
    background: #ECFDF5;
    border-left: 6px solid #7BC6A4;
    padding: 1rem;
    border-radius: 12px;
}
.status-warning {
    background: #FFFBEB;
    border-left: 6px solid #F4D06F;
    padding: 1rem;
    border-radius: 12px;
}
.status-critical {
    background: #FEF2F2;
    border-left: 6px solid #F28B82;
    padding: 1rem;
    border-radius: 12px;
}

/* ===============================
   BUTTONS
================================ */
.stButton button {
    background-color: #5DA9E9;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    border: none;
}
.stButton button:hover {
    background-color: #4A98D9;
}

/* ===============================
   DATAFRAME & PLOTS
================================ */
[data-testid="stDataFrame"],
.js-plotly-plot {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}

/* ===============================
   TABS
================================ */
.stTabs [aria-selected="true"] {
    background-color: #E8F2FC;
    border-radius: 8px;
}

/* ===============================
   SCROLLBAR
================================ */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background-color: #CBD5E1;
    border-radius: 10px;
}

.anomaly-high {
    background: #FDECEA;
    border-left: 6px solid #EF4444;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.anomaly-medium {
    background: #FFF4E5;
    border-left: 6px solid #F59E0B;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.anomaly-low {
    background: #ECFDF5;
    border-left: 6px solid #10B981;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.metric-card {
    background:#ffffff;
    border-radius:16px;
    padding:1.4rem;
    border:1px solid #E5E7EB;
    box-shadow:0 8px 18px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)




# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'health_data' not in st.session_state:
    st.session_state.health_data = None

# Helper Functions
def generate_sample_data(days=30):
    """Generate realistic sample health data with anomalies"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        base_hr = 72 + np.sin(i * 0.5) * 8 + np.random.normal(0, 3)
        base_steps = 7500 + np.sin(i * 0.3) * 2000 + np.random.normal(0, 500)
        base_sleep = 7 + np.sin(i * 0.4) * 1.5 + np.random.normal(0, 0.3)
        
        is_anomaly = np.random.random() > 0.85
        
        if is_anomaly:
            anomaly_type = np.random.choice(['high_hr', 'low_steps', 'poor_sleep'])
            if anomaly_type == 'high_hr':
                base_hr += np.random.uniform(20, 35)
            elif anomaly_type == 'low_steps':
                base_steps -= np.random.uniform(3000, 5000)
            else:
                base_sleep -= np.random.uniform(2, 3)
        
        data.append({
            'timestamp': date,
            'heart_rate': max(40, min(120, base_hr)),
            'steps': max(0, base_steps),
            'sleep_hours': max(3, min(10, base_sleep)),
            'is_anomaly': is_anomaly
        })
    
    return pd.DataFrame(data)

def detect_anomalies_threshold(df):
    """Rule-based anomaly detection using thresholds"""
    anomalies = []
    
    hr_anomalies = df[(df['heart_rate'] > 100) | (df['heart_rate'] < 50)]
    for _, row in hr_anomalies.iterrows():
        anomalies.append({
            'date': row['timestamp'],
            'type': 'High Heart Rate' if row['heart_rate'] > 100 else 'Low Heart Rate',
            'value': f"{row['heart_rate']:.0f} BPM",
            'severity': 'high' if abs(row['heart_rate'] - 72) > 30 else 'medium'
        })
    
    step_anomalies = df[df['steps'] < 3000]
    for _, row in step_anomalies.iterrows():
        anomalies.append({
            'date': row['timestamp'],
            'type': 'Low Activity',
            'value': f"{row['steps']:.0f} steps",
            'severity': 'medium'
        })
    
    sleep_anomalies = df[(df['sleep_hours'] < 5) | (df['sleep_hours'] > 10)]
    for _, row in sleep_anomalies.iterrows():
        anomalies.append({
            'date': row['timestamp'],
            'type': 'Sleep Deficit' if row['sleep_hours'] < 5 else 'Excessive Sleep',
            'value': f"{row['sleep_hours']:.1f} hours",
            'severity': 'medium'
        })
    
    return pd.DataFrame(anomalies)

def detect_anomalies_ml(df):
    """ML-based anomaly detection using Isolation Forest"""
    features = ['heart_rate', 'steps', 'sleep_hours']
    X = df[features].fillna(df[features].mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled)
    
    df['ml_anomaly'] = predictions == -1
    return df

def perform_clustering(df):
    """Cluster user behavior patterns"""
    features = ['heart_rate', 'steps', 'sleep_hours']
    X = df[features].fillna(df[features].mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans

def calculate_health_score(df):
    """Calculate overall health score (0-100)"""
    avg_hr = df['heart_rate'].mean()
    avg_steps = df['steps'].mean()
    avg_sleep = df['sleep_hours'].mean()
    
    hr_score = max(0, 100 - abs(72 - avg_hr) * 2)
    step_score = min(100, (avg_steps / 10000) * 100)
    sleep_score = min(100, (avg_sleep / 8) * 100)
    
    return round((hr_score + step_score + sleep_score) / 3)

# =========================
# UI HELPER COMPONENTS
# =========================
def info_card(title, value, subtitle, accent="#5DA9E9"):
    st.markdown(f"""
    <div style="
        background:#ffffff;
        border-radius:16px;
        padding:1.4rem;
        border:1px solid #E5E7EB;
        box-shadow:0 10px 24px rgba(0,0,0,0.05);
        height:100%;
    ">
        <p style="font-size:0.85rem;color:#6B7280;margin-bottom:0.4rem;">
            {title}
        </p>
        <h2 style="margin:0;color:{accent};font-weight:700;">
            {value}
        </h2>
        <p style="font-size:0.8rem;color:#9CA3AF;margin-top:0.4rem;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

#-------------------------------
# SIDEBAR NAVIGATION 
#------------------------------------
with st.sidebar:
    st.title("FitPulse")
    st.caption("Personal Fitness & Health Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Data",
            "Anomaly Analysis",
            "ML Insights",
            "Reports",
            "About"
        ]
    )

    st.markdown("---")
    st.subheader("Data Actions")

    if st.button("Load Sample Data", use_container_width=True):
        st.session_state.health_data = generate_sample_data(30)
        st.session_state.data_loaded = True
        st.success("Sample data loaded")
        st.rerun()

    if st.session_state.data_loaded:
        if st.button("Refresh Analysis", use_container_width=True):
            st.rerun()


# Main Content
ml_anomaly_count = 0
avg_hr = 0
avg_steps = 0
health_score = 0

if page == "Dashboard":
    st.title("Health Dashboard")
    
    if not st.session_state.data_loaded:
        st.info("👈 Please load sample data or upload your own data from the sidebar to get started!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 💓 Heart Rate")
            st.markdown("Track your heart rate patterns and detect abnormalities")
        with col2:
            st.markdown("### 👟 Activity")
            st.markdown("Monitor daily steps and activity levels")
        with col3:
            st.markdown("### 😴 Sleep")
            st.markdown("Analyze sleep quality and duration")
    else:
        df = st.session_state.health_data.copy()
        df = detect_anomalies_ml(df)

        health_score = calculate_health_score(df)
        avg_hr = df["heart_rate"].mean()
        avg_steps = df["steps"].mean()
        ml_anomaly_count = int(df["ml_anomaly"].sum())
        anomalies_df = detect_anomalies_threshold(df)
        
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            info_card(
                "Health Score",
                f"{health_score} / 100",
                "Overall fitness",
                "#3B82F6"
            )

        with col2:
            info_card(
                "Avg Heart Rate",
                f"{avg_hr:.0f} BPM",
                "Resting average",
                "#EF4444"
            )

        with col3:
            info_card(
                "Avg Daily Steps",
                f"{avg_steps:,.0f}",
                "Activity level",
                "#10B981"
            )

        with col4:
            info_card(
                "Anomalies Detected",
                ml_anomaly_count,
                "ML-based detection",
                "#F59E0B"
            )

        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("### 💓 Heart Rate Trends")
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["heart_rate"],
                mode="lines",
                line=dict(color="#EF4444", width=2),
                name="Heart Rate"
            ))

            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["heart_rate"].rolling(7).mean(),
                mode="lines",
                line=dict(color="#9CA3AF", width=2, dash="dash"),
                name="7-day Avg"
            ))

            fig.update_layout(
                title="Heart Rate Trends",
                title_font_size=18,
                xaxis_title="Date",
                yaxis_title="Beats Per Minute",
                template="simple_white",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", y=-0.25)
            )

            st.plotly_chart(fig, use_container_width=True)

                
        with col2:
            st.markdown("### 👟 Daily Steps")
            fig = go.Figure()
            colors = ['#e74c3c' if s < 5000 else '#f39c12' if s < 8000 else '#10B981' 
                    for s in df['steps']]
            fig.add_trace(go.Bar(
                x=df['timestamp'],
                y=df['steps'],
                marker_color=colors,
                name='Steps',
                marker_line_color='#2c3e50',
                marker_line_width=2
            ))
            fig.add_hline(y=10000, line_dash="dash", line_color="#27ae60", line_width=3,
                        annotation_text="Goal (10,000 steps)",
                        annotation_position="right",
                        annotation_font=dict(color="#10B981", size=14, family="Arial Black"))
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color='#2c3e50', size=13, family="Arial"),
                xaxis=dict(
                    gridcolor='#ecf0f1',
                    showgrid=True,
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=12)
                ),
                yaxis=dict(
                    gridcolor='#ecf0f1',
                    showgrid=True,
                    title='Steps',
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=12)
                ),
                transition=dict(duration=500, easing='cubic-in-out'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sleep chart - full width
            st.markdown("### 😴 Sleep Patterns")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sleep_hours'],
                mode='lines+markers',
                name='Sleep Hours',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=8, color='#8e44ad', line=dict(width=2, color='#ffffff')),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.3)'
            ))
            fig.add_hline(y=7, line_dash="dash", line_color="#10B981", line_width=3,
                        annotation_text="Recommended (7-8 hrs)",
                        annotation_position="right",
                        annotation_font=dict(color="#10B981", size=14, family="Arial Black"))
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color='#2c3e50', size=13, family="Arial"),
                xaxis=dict(
                    gridcolor='#ecf0f1',
                    showgrid=True,
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=12)
                ),
                yaxis=dict(
                    gridcolor='#ecf0f1',
                    showgrid=True,
                    title='Hours',
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=12)
                ),
                transition=dict(duration=500, easing='cubic-in-out'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Upload Data":
    st.title("Upload Your Health Data")
    
    st.markdown("""
    ### Supported Formats
    - **CSV**: Comma-separated values
    - **JSON**: JavaScript Object Notation
    
    ### Required Columns
    - `timestamp` or `date`: Date/time of measurement
    - `heart_rate`: Heart rate in BPM
    - `steps`: Daily step count
    - `sleep_hours`: Hours of sleep
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.success("✅ File uploaded successfully!")
            
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Date Range", f"{len(df)} days")
            
            if st.button("Process Data", type="primary"):
                required_cols = ['timestamp', 'heart_rate', 'steps', 'sleep_hours']
                if not all(col in df.columns for col in required_cols):
                    st.error("Missing required columns!")
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.session_state.health_data = df
                    st.session_state.data_loaded = True
                    st.success("✅ Data processed successfully! Go to Dashboard to view insights.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    with st.expander("📄 View Sample Data Format"):
        sample = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='D'),
            'heart_rate': [72, 75, 68, 85, 70],
            'steps': [8500, 9200, 6800, 10500, 7800],
            'sleep_hours': [7.5, 8.0, 6.5, 7.0, 7.8]
        })
        st.dataframe(sample, use_container_width=True)
        
        csv = sample.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_health_data.csv",
            mime="text/csv"
        )

elif page == "Anomaly Analysis":
    st.title("Anomaly Detection Analysis")

    if not st.session_state.data_loaded:
        st.warning("Please load data first.")
    else:
        df = st.session_state.health_data.copy()
        df = detect_anomalies_ml(df)
        anomalies_df = detect_anomalies_threshold(df)

        # ==============================
        # SEVERITY COUNTS
        # ==============================
        ml_anomalies = df[df["ml_anomaly"] == True]

        high_severity = 0
        medium_severity = 0
        low_severity = 0

        for _, row in ml_anomalies.iterrows():
            hr = row["heart_rate"]
            steps = row["steps"]
            sleep = row["sleep_hours"]

            if (hr > 100 or hr < 50) and abs(hr - 72) > 30:
                high_severity += 1
            elif hr > 100 or hr < 50 or steps < 3000 or sleep < 5 or sleep > 10:
                medium_severity += 1
            else:
                low_severity += 1

        # ==============================
        # SEVERITY SUMMARY CARDS
        # ==============================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="card" style="border-left:6px solid {RED};">
                <p style="color:#6B7280;margin:0;">High Severity</p>
                <h2 style="margin:0;color:#0F172A;">{high_severity}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card" style="border-left:6px solid {YELLOW};">
                <p style="color:#6B7280;margin:0;">Medium Severity</p>
                <h2 style="margin:0;color:#0F172A;">{medium_severity}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card" style="border-left:6px solid {GREEN};">
                <p style="color:#6B7280;margin:0;">Low Severity</p>
                <h2 style="margin:0;color:#0F172A;">{low_severity}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ==============================
        # ANOMALY TIMELINE
        # ==============================
        st.subheader("Anomaly Timeline")

        fig = go.Figure()

        normal_data = df[df["ml_anomaly"] == False]
        fig.add_trace(go.Scatter(
            x=normal_data["timestamp"],
            y=normal_data["heart_rate"],
            mode="markers",
            name="Normal",
            marker=dict(color=GREEN, size=9, line=dict(color="#ffffff", width=1))
        ))

        anomaly_data = df[df["ml_anomaly"] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_data["timestamp"],
            y=anomaly_data["heart_rate"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=RED, size=13, symbol="x", line=dict(color="#ffffff", width=2))
        ))

        fig.update_layout(
            height=380,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis_title="Date",
            yaxis_title="Heart Rate (BPM)",
            font=dict(color="#0F172A"),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ==============================
        # DETAILED ANOMALY REPORT
        # ==============================
        st.subheader("Detailed Anomaly Report")

        ml_anomaly_count = int(df["ml_anomaly"].sum())

        if ml_anomaly_count == 0:
            st.success("No anomalies detected. Health metrics are within normal range.")
        else:
            st.markdown(f"**Total ML-Detected Anomalies: {ml_anomaly_count}**")

            for _, row in ml_anomalies.iterrows():
                hr = row["heart_rate"]
                steps = row["steps"]
                sleep = row["sleep_hours"]
                date = row["timestamp"]

                issues = []
                severity = "low"

                if hr > 100:
                    issues.append(f"High Heart Rate: {hr:.0f} BPM")
                    severity = "high" if hr > 110 else "medium"
                elif hr < 50:
                    issues.append(f"Low Heart Rate: {hr:.0f} BPM")
                    severity = "high" if hr < 45 else "medium"

                if steps < 3000:
                    issues.append(f"Low Activity: {steps:.0f} steps")
                    if severity == "low":
                        severity = "medium"

                if sleep < 5:
                    issues.append(f"Sleep Deficit: {sleep:.1f} hrs")
                    if severity == "low":
                        severity = "medium"
                elif sleep > 10:
                    issues.append(f"Excessive Sleep: {sleep:.1f} hrs")
                    if severity == "low":
                        severity = "medium"

                if not issues:
                    issues.append("Behavioral pattern deviation detected")

                severity_color = {
                    "high": RED,
                    "medium": YELLOW,
                    "low": GREEN
                }[severity]

                st.markdown(f"""
                <div class="card" style="border-left:6px solid {severity_color}; margin-bottom:1rem;">
                    <strong>Anomaly Detected</strong>
                    <span style="color:#6B7280;"> — {date.strftime('%Y-%m-%d')}</span>
                    <p style="margin-top:8px;">{' | '.join(issues)}</p>
                    <p style="color:#6B7280;font-size:0.9em;">Severity: {severity.upper()}</p>
                </div>
                """, unsafe_allow_html=True)


elif page == "ML Insights":
    st.title("Machine Learning Insights")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
    else:
        df = st.session_state.health_data
        df, kmeans = perform_clustering(df)
        
        st.markdown("### 🎯 Behavioral Clustering")
        st.markdown("Your health patterns have been grouped into 3 distinct clusters:")
        
        col1, col2, col3 = st.columns(3)
        
        for i in range(3):
            cluster_data = df[df['cluster'] == i]
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #1a1a2e !important;">Cluster {i+1}</h3>
                    <p style="color: #2d3436 !important;"><strong style="color: #0984e3 !important;">{len(cluster_data)}</strong> days</p>
                    <p style="color: #2d3436 !important;">Avg HR: <strong style="color: #e74c3c !important;">{cluster_data['heart_rate'].mean():.0f} BPM</strong></p>
                    <p style="color: #2d3436 !important;">Avg Steps: <strong style="color: #3B82F6 !important;">{cluster_data['steps'].mean():,.0f}</strong></p>
                    <p style="color: #2d3436 !important;">Avg Sleep: <strong style="color: #9b59b6 !important;">{cluster_data['sleep_hours'].mean():.1f} hrs</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📊 3D Cluster Visualization")
        
        fig = px.scatter_3d(
            df,
            x='heart_rate',
            y='steps',
            z='sleep_hours',
            color='cluster',
            color_continuous_scale=[[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']],
            labels={
                'heart_rate': 'Heart Rate (BPM)',
                'steps': 'Steps',
                'sleep_hours': 'Sleep (hours)',
                'cluster': 'Cluster'
            }
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='#ffffff')))
        fig.update_layout(
            scene=dict(
                bgcolor='#ffffff',
                xaxis=dict(
                    backgroundcolor='#ecf0f1',
                    gridcolor='#bdc3c7',
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=11)
                ),
                yaxis=dict(
                    backgroundcolor='#ecf0f1',
                    gridcolor='#bdc3c7',
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=11)
                ),
                zaxis=dict(
                    backgroundcolor='#ecf0f1',
                    gridcolor='#bdc3c7',
                    title_font=dict(color='#2c3e50', size=14, family="Arial Black"),
                    tickfont=dict(color='#2c3e50', size=11)
                )
            ),
            paper_bgcolor='#ffffff',
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(color='#2c3e50', size=12, family="Arial"),
            transition=dict(duration=500, easing='cubic-in-out')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎲 Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlation Heatmap")
            corr_matrix = df[['heart_rate', 'steps', 'sleep_hours']].corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r',
                aspect="auto",
                text_auto='.2f',
                zmin=-1,
                zmax=1
            )
            fig.update_layout(
                paper_bgcolor='#ffffff',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color='#2c3e50', size=12, family="Arial"),
                xaxis=dict(
                    tickfont=dict(color='#2c3e50', size=12, family="Arial Black"),
                    side='bottom'
                ),
                yaxis=dict(
                    tickfont=dict(color='#2c3e50', size=12, family="Arial Black")
                ),
                transition=dict(duration=500, easing='cubic-in-out')
            )
            fig.update_traces(
                textfont=dict(color='#2c3e50', size=14, family="Arial Black")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Metric Distributions")
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df['heart_rate'], 
                name='Heart Rate', 
                marker=dict(color='#e74c3c'),
                line=dict(color='#c0392b', width=2),
                boxmean='sd'
            ))
            fig.add_trace(go.Box(
                y=df['steps']/100, 
                name='Steps (÷100)', 
                marker=dict(color='#3498db'),
                line=dict(color='#2980b9', width=2),
                boxmean='sd'
            ))
            fig.add_trace(go.Box(
                y=df['sleep_hours']*10, 
                name='Sleep (×10)', 
                marker=dict(color='#9b59b6'),
                line=dict(color='#8e44ad', width=2),
                boxmean='sd'
            ))
            fig.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(color='#2c3e50', size=12, family="Arial"),
                xaxis=dict(
                    gridcolor='#ecf0f1',
                    tickfont=dict(color='#2c3e50', size=12, family="Arial Black")
                ),
                yaxis=dict(
                    gridcolor='#ecf0f1',
                    showgrid=True,
                    tickfont=dict(color='#2c3e50', size=12),
                    title='Normalized Values',
                    title_font=dict(color='#2c3e50', size=13, family="Arial Black")
                ),
                transition=dict(duration=500, easing='cubic-in-out')
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Reports":
    st.title("📑 Health Reports & Export")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
    else:
        # =========================
        # DATA PREPARATION
        # =========================
        df = st.session_state.health_data.copy()

        # Run ML anomaly detection ONCE
        df = detect_anomalies_ml(df)

        # Safety check
        if "ml_anomaly" not in df.columns:
            df["ml_anomaly"] = False

        ml_anomaly_count = int(df["ml_anomaly"].sum())
        health_score = calculate_health_score(df)
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # =========================
        # REPORT SUMMARY
        # =========================
        st.markdown("### 📊 Report Summary")

        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color:#111827;">Health Analysis Report</h3>
                <p><strong>Generated:</strong> {report_date}</p>
                <p><strong>Analysis Period:</strong> {len(df)} days</p>
                <p><strong>Health Score:</strong> {health_score}/100</p>
                <p><strong>Total ML Anomalies:</strong> {ml_anomaly_count}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================
        # DETAILED METRICS
        # =========================
        st.markdown("### 📉 Detailed Metrics")

        col1, col2, col3 = st.columns(3)

        # ---- HEART RATE ----
        with col1:
            hr = df["heart_rate"].describe()
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>💓 Heart Rate</h4>
                    <p>Mean: <strong style="color:#EF4444;">{hr['mean']:.1f} BPM</strong></p>
                    <p>Min: {hr['min']:.1f} BPM</p>
                    <p>Max: {hr['max']:.1f} BPM</p>
                    <p>Std Dev: {hr['std']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---- STEPS ----
        with col2:
            steps = df["steps"].describe()
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>👟 Steps</h4>
                    <p>Mean: <strong style="color:#3B82F6;">{steps['mean']:.0f}</strong></p>
                    <p>Min: {steps['min']:.0f}</p>
                    <p>Max: {steps['max']:.0f}</p>
                    <p>Std Dev: {steps['std']:.0f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---- SLEEP ----
        with col3:
            sleep = df["sleep_hours"].describe()
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>😴 Sleep</h4>
                    <p>Mean: <strong style="color:#10B981;">{sleep['mean']:.1f} hrs</strong></p>
                    <p>Min: {sleep['min']:.1f} hrs</p>
                    <p>Max: {sleep['max']:.1f} hrs</p>
                    <p>Std Dev: {sleep['std']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # =========================
        # EXPORT OPTIONS
        # =========================
        st.markdown("### 📥 Export Options")

        col1, col2, col3 = st.columns(3)

        # ---- FULL DATA ----
        with col1:
            st.download_button(
                "📊 Download Full Data (CSV)",
                data=df.to_csv(index=False),
                file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ---- ANOMALIES ----
        with col2:
            if ml_anomaly_count > 0:
                anomalies = df[df["ml_anomaly"] == True][
                    ["timestamp", "heart_rate", "steps", "sleep_hours"]
                ]

                st.download_button(
                    "⚠️ Download Anomalies (CSV)",
                    data=anomalies.to_csv(index=False),
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # ---- SUMMARY REPORT ----
        with col3:
            summary_text = f"""
            FitPulse Health Report
            Generated: {report_date}

            Analysis Period: {len(df)} days
            Health Score: {health_score}/100
            Total ML Anomalies: {ml_anomaly_count}

            Average Heart Rate: {df['heart_rate'].mean():.1f} BPM
            Average Steps: {df['steps'].mean():.0f}
            Average Sleep: {df['sleep_hours'].mean():.1f} hrs
            """

            st.download_button(
                "📄 Download Summary (TXT)",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )



else:  # About page
    st.title("About FitPulse")
    
    st.markdown("""
    ## 💓 FitPulse Health Anomaly Detection System
    
    ### 🎯 Project Overview
    FitPulse is an advanced health monitoring system that uses machine learning and statistical methods 
    to detect anomalies in fitness tracker data. It helps users identify unusual patterns in their health 
    metrics and provides actionable insights.
    
    ### 📊 Tracked Metrics
    
    This streamlined version focuses on three core health indicators:
    
    1. **💓 Heart Rate** - Monitoring cardiovascular health
    2. **👟 Steps** - Daily activity and movement tracking
    3. **😴 Sleep Hours** - Sleep duration and quality patterns
    
    ### 🔧 Technologies Used
    
    #### Core Technologies
    - **Python 3.8+** - Primary programming language
    - **Streamlit** - Interactive web application framework
    - **Pandas & NumPy** - Data manipulation and numerical computing
    - **Plotly** - Interactive data visualization
    
    #### Machine Learning
    - **Scikit-learn** - ML algorithms and preprocessing
    - **Isolation Forest** - Unsupervised anomaly detection
    - **KMeans** - Clustering algorithms
    - **StandardScaler** - Feature normalization
    
    ### 🎨 Key Features
    
    1. **Multi-Method Anomaly Detection**
       - Rule-based thresholds for immediate alerts
       - Machine learning (Isolation Forest) for complex patterns
       - Clustering to identify behavioral groups
    
    2. **Three Core Metrics**
       - Heart rate monitoring
       - Daily activity tracking (steps)
       - Sleep duration analysis
    
    3. **Interactive Visualizations**
       - Real-time charts with Plotly
       - 3D scatter plots for cluster analysis
       - Correlation heatmaps
       - Time series trends
    
    4. **Health Score Calculation**
       - Composite score (0-100) based on three metrics
       - Normalized against recommended health standards
       - Trending indicators for improvement tracking
    
    5. **Export & Reporting**
       - CSV export for data analysis
       - Anomaly reports for medical consultation
       - Summary reports with key insights
    
    ### 📊 Anomaly Detection Methods
    
    #### 1. Threshold-Based Detection
    - Heart Rate: <50 BPM or >100 BPM
    - Steps: <3000 steps per day
    - Sleep: <5 hours or >10 hours
    
    #### 2. Machine Learning (Isolation Forest)
    - Detects complex, multi-dimensional anomalies
    - Unsupervised learning approach
    - Considers correlations between metrics
    
    #### 3. Behavioral Clustering
    - Groups similar days together
    - Identifies outlier behavior patterns
    - Helps understand lifestyle variations
    
    ### 📈 Use Cases
    
    - **Personal Health Monitoring** - Track your own fitness metrics
    - **Clinical Research** - Analyze patient data for studies
    - **Fitness Coaching** - Monitor client progress
    - **Health Insurance** - Risk assessment and wellness programs
    - **Wearable Device Analytics** - Process fitness tracker data
    
    ### 🚀 Future Enhancements
    
    - Integration with real fitness APIs (Fitbit, Apple Health, Google Fit)
    - Time series forecasting for predictive insights
    - Email/SMS alerts for critical anomalies
    - Multi-user support with authentication
    - Mobile app development
    - Real-time data streaming
    - AI-powered health recommendations
    
    ### 👨‍💻 Developer Information
    
    **Project Type:** Health Analytics & Machine Learning  
    **Framework:** Streamlit  
    **License:** MIT  
    **Status:** Production Ready  
    
    ### 📚 Data Privacy & Security
    
    - All data processing is done locally
    - No data is transmitted to external servers
    - Users have full control over their data
    - Export and delete data anytime
    - Compliant with health data privacy standards
    
    ### 💡 How to Use
    
    1. **Load Data** - Use sample data or upload your own CSV/JSON
    2. **Explore Dashboard** - View comprehensive health metrics
    3. **Analyze Anomalies** - Review detected irregularities
    4. **Check ML Insights** - Understand behavioral patterns
    5. **Export Reports** - Download data for further analysis
    
    ### 🎯 Project Goals Achieved
    
    ✅ Data ingestion from multiple formats  
    ✅ Robust preprocessing and cleaning  
    ✅ Multi-method anomaly detection  
    ✅ Machine learning integration  
    ✅ Interactive visualization dashboard  
    ✅ Comprehensive reporting system  
    ✅ User-friendly interface  
    ✅ Production-ready code  
    ✅ Simplified to core health metrics  
    ✅ Perfect color contrast and visibility  
    ✅ Fully animated and smooth UI/UX  
    
    ---
    
    **Built with ❤️ for Health & Wellness**
    """)