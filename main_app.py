import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.io as pio
pio.templates.default = "plotly_dark"
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from theme import apply_fitpulse_plotly_theme

if "health_data" not in st.session_state:
    st.session_state.health_data = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# THEME CONSTANTS #


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
   GLOBAL THEME (DARK)
================================ */
html, body {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0F172A;
}

.stApp {
    background-color: #0F172A;
    color: #ffffff;
}

/* ===============================
   PAGE LAYOUT
================================ */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===============================
   HEADINGS
================================ */
h1, h2, h3, h4 {
    color: #ffffff;
    font-weight: 700;
}

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #1F2937;
}
section[data-testid="stSidebar"] .sidebar-title {
    color: #7DD3FC !important;
    font-weight: 700;
}

.section-title {
    color: #7DD3FC;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}

            
/* ===============================
   NAV BUTTONS
================================ */
.nav-card {
    padding: 10px 14px;
    margin-bottom: 10px;
    border-radius: 8px;

    /* Navy gradient like sidebar */
    background: linear-gradient(180deg, #020617, #0F172A);

    color: #CBD5E1;
    font-weight: 500;
    text-align: center;

    /* Blue gradient outline */
    border: 1.5px solid transparent;
    background-clip: padding-box;
    box-shadow: 0 0 0 1.5px #38BDF8;

    cursor: pointer;

    transition: 
        box-shadow 0.2s ease,
        background 0.2s ease,
        color 0.2s ease;
}

/* Hover → white outline */
.nav-card:hover {
    box-shadow: 0 0 0 1.5px #FFFFFF;
}

/* Active tab → blue gradient fill */
.nav-active {
    background: linear-gradient(90deg, #38BDF8, #2563EB);
    color: #FFFFFF;
    font-weight: 600;

    box-shadow: none;
    border: none;
}

/* ===============================
   METRIC & INFO CARDS
================================ */
.metric-card,
[data-testid="stMetric"],
.card {
    background: linear-gradient(
        145deg,
        #020617,
        #111827
    );
    border-radius: 18px;
    padding: 1.4rem;
    border: 1px solid #1F2937;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    transition: 0.3s ease;
}


/* Metric text */
[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

[data-testid="stMetricValue"] {
    color: #38BDF8;
    font-size: 2.1rem;
    font-weight: 800;
}

/* ===============================
   STATUS / ANOMALY CARDS
================================ */
.anomaly-high {
    background: rgba(239, 68, 68, 0.15);
    border: 6px solid #EF4444;
}

.anomaly-medium {
    background: rgba(250, 204, 21, 0.15);
    border: 6px solid #FACC15;
}

.anomaly-low {
    background: rgba(34, 197, 94, 0.15);
    border: 6px solid #22C55E;
}
/* ===============================
   BUTTONS (MINIMAL NAV STYLE)
================================ */
.stButton button {
    background: linear-gradient(180deg, #020617, #0F172A);
    color: #CBD5E1;
    font-weight: 500;

    border-radius: 10px;
    padding: 0.55rem 1.4rem;

    /* Blue outline */
    border: 1.5px solid #38BDF8;

    box-shadow: none;
    transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
}

/* Hover → white border */
.stButton button:hover {
    border-color: #FFFFFF;
    background: linear-gradient(180deg, #020617, #0F172A);
}

/* Active tab (clicked / selected) */
.stButton button:focus,
.stButton button:active {
    background: linear-gradient(90deg, #38BDF8, #2563EB);
    color: #FFFFFF;
    border-color: transparent;
    box-shadow: none;
}


/* ===============================
   DOWNLOAD BUTTONS
================================ */
div.stDownloadButton > button {
    background: linear-gradient(135deg, #020617, #0F172A);
    color: #E5E7EB;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.6rem 1.6rem;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(56,189,248,0.25);
    transition: all 0.3s ease;
}

div.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #020617, #1E293B);
    color: #38BDF8;
    border-color: #38BDF8;
    box-shadow: 0 0 25px rgba(56,189,248,0.5);
    transform: translateY(-2px);
}


/* ===============================
   DATAFRAME & PLOTS
================================ */
[data-testid="stDataFrame"],
.js-plotly-plot {       
    background-color: #020617 !important;
    border-radius: 18px;
    border: 1px solid #1F2937;
}

.fitpulse-caption {
    font-size: 1rem;
    color: #ffffff;
    text-align: center;
    margin-top: -6px;
    margin-bottom: 14px;
    font-style: italic;

    background: rgba(148,163,184,0.08);
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.15);
} 
/* ===============================
   TABS
================================ */
.stTabs [aria-selected="true"] {
    background-color: #020617;
    border-radius: 10px;
    border: 1px solid #38BDF8;
}

/* ===============================
   SCROLLBAR
================================ */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(#38BDF8, #2563EB);
    border-radius: 10px;
}

/* Reduce spacing around status badges inside cards */
div[data-testid="stAlert"] {
    margin-top: 6px;
    margin-bottom: 8px;
    padding: 10px 14px;
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

def generate_pdf_report(df, health_score, ml_anomaly_count, risk_label):
    file_path = os.path.join(os.getcwd(), "FitPulse_Health_Report.pdf")

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    # =========================
    # TITLE
    # =========================
    content.append(Paragraph("<b>FitPulse Health Report</b>", styles["Title"]))
    content.append(Spacer(1, 16))

    # =========================
    # META INFO
    # =========================
    content.append(Paragraph(
        f"""
        <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Analysis Period:</b> {len(df)} days<br/>
        <b>Health Score:</b> {health_score}/100<br/>
        <b>Total ML Anomalies:</b> {ml_anomaly_count}<br/>
        <b>Risk Level:</b> {risk_label}
        """,
        styles["Normal"]
    ))
    content.append(Spacer(1, 14))

    # =========================
    # AVERAGE METRICS
    # =========================
    content.append(Paragraph("<b>Average Health Metrics</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(
        f"""
        • <b>Heart Rate:</b> {df['heart_rate'].mean():.1f} BPM<br/>
        • <b>Daily Steps:</b> {df['steps'].mean():.0f}<br/>
        • <b>Sleep Duration:</b> {df['sleep_hours'].mean():.1f} hrs
        """,
        styles["Normal"]
    ))
    content.append(Spacer(1, 14))

    # =========================
    # METRIC RANGES
    # =========================
    content.append(Paragraph("<b>Observed Metric Ranges</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(
        f"""
        <b>Heart Rate:</b> {df['heart_rate'].min():.0f} – {df['heart_rate'].max():.0f} BPM<br/>
        <b>Steps:</b> {df['steps'].min():.0f} – {df['steps'].max():.0f}<br/>
        <b>Sleep:</b> {df['sleep_hours'].min():.1f} – {df['sleep_hours'].max():.1f} hrs
        """,
        styles["Normal"]
    ))
    content.append(Spacer(1, 14))

    # =========================
    # DOCTOR NOTES
    # =========================
    content.append(Paragraph("<b>Clinical Notes</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))

    note = "Patient metrics are within acceptable ranges."
    if risk_label == "HIGH":
        note = "High-risk indicators detected. Further medical evaluation recommended."
    elif risk_label == "MEDIUM":
        note = "Moderate deviations observed. Lifestyle adjustments advised."

    content.append(Paragraph(note, styles["Normal"]))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(content)

    return file_path



def get_risk_level(health_score, anomaly_count):
    if health_score >= 80 and anomaly_count == 0:
        return "LOW", "#22C55E"
    elif health_score >= 60 and anomaly_count <= 3:
        return "MEDIUM", "#FACC15"
    else:
        return "HIGH", "#EF4444"


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

def status_badge(label, level):
    if level == "good":
        st.success(f"🟢 {label}")
    elif level == "moderate":
        st.warning(f"🟡 {label}")
    else:
        st.error(f"🔴 {label}")


def chart_card_start():
    st.markdown("""
    <div style="
        background:#020617;
        padding:16px;
        border-radius:14px;
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:16px;
    ">
    """, unsafe_allow_html=True)


def chart_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


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
# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown(
    """
    <h2 class="sidebar-title">🩺 FitPulse</h2>
    """,
    unsafe_allow_html=True
)


    def nav_button(label, page_name):
        if st.session_state.page == page_name:
            st.markdown(
                f"<div class='nav-card nav-active'>{label}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.button(label, use_container_width=True, key=f"nav_{page_name}"):
                st.session_state.page = page_name
                st.rerun()

    # -------- NAVIGATION --------
    nav_button("Home", "Dashboard")
    nav_button("Upload Data", "Upload Data")
    nav_button("Anomaly Analysis", "Anomaly Analysis")
    nav_button("ML Insights", "ML Insights")
    nav_button("Reports", "Reports")

    st.markdown("---")
    st.markdown(
    "<h3 style='color:#ffffff' class='section-title'>⚙️ Data Actions</h3>",
    unsafe_allow_html=True
    )


    # -------- DATA ACTIONS --------
    if st.button("Load Sample Data", use_container_width=True, key="load_sample"):
        st.session_state.health_data = generate_sample_data(30)
        st.session_state.data_loaded = True
        st.success("Sample data loaded successfully")
        st.rerun()

    if st.session_state.data_loaded:
        if st.button("Refresh Analysis", use_container_width=True, key="refresh_analysis"):
            # Force recalculation by running anomaly detection again
            if st.session_state.health_data is not None:
                st.session_state.health_data = detect_anomalies_ml(st.session_state.health_data)
                st.success("Analysis refreshed successfully!")
                st.rerun()

# -----------------------------
# Main Content

page = st.session_state.page
ml_anomaly_count = 0
avg_hr = 0
avg_steps = 0
health_score = 0


if page == "Dashboard":
    st.markdown("""
    <h2 style="color:#38BDF8;text-shadow:0 0 12px rgba(56,189,248,0.6);">
    💓 Health Dashboard
    </h2>
    """, unsafe_allow_html=True)
    
    
    if not st.session_state.data_loaded:
        st.info("👈 Please load sample data or upload your own data from the sidebar to get started!")
        st.info(
            "FitPulse analyzes your activity, sleep, and heart rate patterns using machine learning "
            "and groups them into easy-to-understand health behavior categories. "
            "This helps you quickly recognize trends and take action."
        )

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
            
            chart_card_start()

            st.markdown("### 💓 Heart Rate Trends")
            avg_hr = df["heart_rate"].mean()

            if avg_hr < 75:
                status_badge("Heart rate is stable", "good")
            elif avg_hr < 90:
                status_badge("Heart rate is moderately elevated", "moderate")
            else:
                status_badge("High heart rate detected", "risk")

            st.caption("Daily heart rate with 7-day moving average")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["heart_rate"],
                mode="lines",
                line=dict(color="#EF4444", width=2),
                name="Daily HR"
            ))

            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["heart_rate"].rolling(7).mean(),
                mode="lines",
                line=dict(color="#9CA3AF", width=2, dash="dash"),
                name="7-day Avg"
            ))

            fig.update_layout(
                hovermode="x unified",
                margin=dict(l=30, r=30, t=40, b=30)
            )

            fig = apply_fitpulse_plotly_theme(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()


                
        with col2:
            
            chart_card_start()

            st.markdown("### 👟 Daily Steps")

            goal_rate = (df["steps"] >= 10000).mean() * 100

            if goal_rate > 60:
                status_badge("Good activity consistency", "good")
            elif goal_rate > 30:
                status_badge("Moderate activity levels", "moderate")
            else:
                status_badge("Low daily activity detected", "risk")

            st.caption("Bars show daily activity against step goal")

            colors = [
                "#EF4444" if s < 5000 else
                "#FACC15" if s < 8000 else
                "#10B981"
                for s in df["steps"]
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["timestamp"],
                y=df["steps"],
                marker_color=colors,
                name="Steps"
            ))

            fig.add_hline(
                y=10000,
                line_dash="dash",
                line_color="#10B981",
                annotation_text="Goal (10k)",
                annotation_font=dict(color="#10B981")
            )

            fig.update_layout(
                hovermode="x unified",
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=False
            )

            fig = apply_fitpulse_plotly_theme(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
            chart_card_end()

            
        # Sleep chart - full width

        chart_card_start()

        avg_sleep = df["sleep_hours"].mean()

        st.markdown("### 😴 Sleep Patterns")
        if avg_sleep >= 7:
            status_badge("Healthy sleep duration", "good")
        elif avg_sleep >= 6:
            status_badge("Sleep slightly below optimal", "moderate")
        else:
            status_badge("Sleep deficit detected", "risk")
            
        st.caption("Daily sleep duration with recommended baseline")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["sleep_hours"],
            mode="lines+markers",
            line=dict(color="#8B5CF6", width=3),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.25)",
            name="Sleep Hours"
        ))

        fig.add_hline(
            y=7,
            line_dash="dash",
            line_color="#10B981",
            annotation_text="Recommended (7–8 hrs)",
            annotation_font=dict(color="#10B981")
        )

        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=30, r=30, t=40, b=30)
        )

        fig = apply_fitpulse_plotly_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()


elif page == "Upload Data":
    st.markdown("""
    <h2 style="color:#38BDF8;text-shadow:0 0 12px rgba(56,189,248,0.6);">
      Upload Your Health Data
    </h2>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div style="color:#E5E7EB; font-size:15px; line-height:1.6">

    ### Supported Formats
    - **CSV**: Comma-separated values  
    - **JSON**: JavaScript Object Notation  

    ### Required Columns
    - `timestamp` or `date`
    - `heart_rate`
    - `steps`
    - `sleep_hours`

    </div>
    """, unsafe_allow_html=True)

    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            st.markdown("### Data Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

            
            col1, col2, col3 = st.columns(3)
            with col1:
                info_card("Total Rows", len(df), "Records", "#38BDF8")

            with col2:
                info_card("Total Columns", len(df.columns), "Columns", "#38BDF8")

            with col3:
                info_card("Date Range", f"{len(df)} days", "Days", "#38BDF8")
            
            if st.button("Process Data", type="primary"):
                required_cols = ['timestamp', 'heart_rate', 'steps', 'sleep_hours']
                if not all(col in df.columns for col in required_cols):
                    st.error("Missing required columns!")
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.session_state.health_data = df
                    st.session_state.data_loaded = True
                    st.success("Data processed successfully! Go to Dashboard to view insights.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    with st.expander("View Sample Data Format"):
        st.markdown("<div style='color:#E5E7EB'>", unsafe_allow_html=True)
        sample = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='D'),
            'heart_rate': [72, 75, 68, 85, 70],
            'steps': [8500, 9200, 6800, 10500, 7800],
            'sleep_hours': [7.5, 8.0, 6.5, 7.0, 7.8]
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        csv = sample.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_health_data.csv",
            mime="text/csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Anomaly Analysis":
    st.markdown("""
    <h2 style="color:#EF4444;text-shadow:0 0 12px rgba(239,68,68,0.6);">
       Anomaly Detection Analysis
    </h2>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first.")
    else:
        df = st.session_state.health_data.copy()
        df = detect_anomalies_ml(df)

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

        def severity_card(title, value, color):
            st.markdown(f"""
            <div style="
                background:#020617;
                border:6px solid {color};
                padding:16px;
                border-radius:10px;
                box-shadow:0 0 15px rgba(0,0,0,0.4);
            ">
                <p style="color:#9CA3AF;margin:0;">{title}</p>
                <h2 style="margin:0;color:#E5E7EB;">{value}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col1:
            severity_card("High Severity", high_severity, "#EF4444")
        with col2:
            severity_card("Medium Severity", medium_severity, "#F59E0B")
        with col3:
            severity_card("Low Severity", low_severity, "#10B981")

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
            marker=dict(color="#10B981", size=9)
        ))

        anomaly_data = df[df["ml_anomaly"] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_data["timestamp"],
            y=anomaly_data["heart_rate"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#EF4444", size=14, symbol="x")
        ))

        fig = apply_fitpulse_plotly_theme(fig, height=400)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ==============================
        # DETAILED ANOMALY REPORT
        # ==============================
        st.subheader("Detailed Anomaly Report")

        if ml_anomalies.empty:
            st.success("No anomalies detected. Health metrics are within normal range.")
        else:
            st.markdown(f"<p style='color:#E5E7EB;'>Total ML-Detected Anomalies: <strong>{len(ml_anomalies)}</strong></p>",
                        unsafe_allow_html=True)

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
                    severity = "medium" if severity == "low" else severity

                if sleep < 5:
                    issues.append(f"Sleep Deficit: {sleep:.1f} hrs")
                    severity = "medium" if severity == "low" else severity
                elif sleep > 10:
                    issues.append(f"Excessive Sleep: {sleep:.1f} hrs")
                    severity = "medium" if severity == "low" else severity

                color = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}[severity]

                st.markdown(f"""
                <div style="
                    background:#020617;
                    border:6px solid {color};
                    padding:14px;
                    border-radius:10px;
                    margin-bottom:12px;
                ">
                    <strong style="color:#E5E7EB;">Anomaly Detected</strong>
                    <span style="color:#9CA3AF;"> — {date.strftime('%Y-%m-%d')}</span>
                    <p style="margin-top:6px;color:#E5E7EB;">{' | '.join(issues)}</p>
                    <p style="color:#9CA3AF;font-size:0.9em;">Severity: {severity.upper()}</p>
                </div>
                """, unsafe_allow_html=True)



elif page == "ML Insights":
    st.markdown("""
    <h2 style="color:#FACC15;text-shadow:0 0 12px rgba(250,204,21,0.6);">
       Machine Learning Insights
    </h2>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
    else:
        df = st.session_state.health_data.copy()
        df, kmeans = perform_clustering(df)

        st.markdown("Behavioral Clustering")
        st.markdown("<p style='color:#9CA3AF;'>Your health patterns have been grouped into 3 distinct clusters.</p>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        def cluster_card(title, days, hr, steps, sleep):
            st.markdown(f"""
            <div style="
                background:#020617;
                padding:16px;
                border-radius:12px;
                border:1px solid rgba(255,255,255,0.08);
                box-shadow:0 0 15px rgba(0,0,0,0.4);
            ">
                <h3 style="color:#FACC15;margin-bottom:8px;">{title}</h3>
                <p style="color:#9CA3AF;margin:0;">{days} days</p>
                <p style="color:#E5E7EB;">Avg HR: <strong style="color:#EF4444;">{hr:.0f} BPM</strong></p>
                <p style="color:#E5E7EB;">Avg Steps: <strong style="color:#38BDF8;">{steps:,.0f}</strong></p>
                <p style="color:#E5E7EB;">Avg Sleep: <strong style="color:#10B981;">{sleep:.1f} hrs</strong></p>
            </div>
            """, unsafe_allow_html=True)

        for i, col in enumerate([col1, col2, col3]):
            cluster_data = df[df["cluster"] == i]
            with col:
                cluster_card(
                    f"Cluster {i+1}",
                    len(cluster_data),
                    cluster_data["heart_rate"].mean(),
                    cluster_data["steps"].mean(),
                    cluster_data["sleep_hours"].mean()
                )

        st.markdown("---")

        behavior_colors = {
            "Active & Healthy": "#22C55E",   # green
            "Moderate": "#FACC15",           # yellow
            "At Risk": "#EF4444"             # red
        }


        cluster_map = {
            0: "Active & Healthy",
            1: "Moderate",
            2: "At Risk"
        }

        df["behavior_label"] = df["cluster"].map(cluster_map)

        st.markdown("###  Personalized Health Insights")

        total_days = len(df)
        healthy_days = (df["behavior_label"] == "🟢 Active & Healthy").sum()
        moderate_days = (df["behavior_label"] == "🟡 Moderate – Needs Improvement").sum()
        risk_days = (df["behavior_label"] == "🔴 At Risk").sum()

        if risk_days / total_days > 0.4:
            insight = (
                "⚠️ Your recent health patterns show frequent low activity or insufficient sleep. "
                "Consider improving daily movement and maintaining a consistent sleep schedule."
            )
        elif healthy_days / total_days > 0.5:
            insight = (
                "✅ Great job! Most of your recent days reflect healthy activity, good sleep, "
                "and stable heart rate patterns. Keep maintaining this routine."
            )
        else:
            insight = (
                "ℹ️ Your health behavior is mixed. With slightly improved sleep duration or daily activity, "
                "you can move toward a more balanced lifestyle."
            )

        st.success(insight)
        
        # ==============================
        # CLUSTER VISUALIZATION
        # ==============================

        st.markdown("###  Health Behavior Overview")

        fig = px.scatter(
            df,
            x="steps",
            y="sleep_hours",
            size="heart_rate",
            color="behavior_label",
            template="plotly_dark",
            color_discrete_map=behavior_colors,
            category_orders={
                "behavior_label": ["Active & Healthy", "Moderate", "At Risk"]
            },
            labels={
                "steps": "Daily Activity (Steps)",
                "sleep_hours": "Sleep Duration (Hours)",
                "behavior_label": "Health Behavior"
            }
        )
        

        fig.update_traces(
            marker=dict(
                opacity=0.75,
                line=dict(width=1, color="#ffffff")
            )
        )

        fig = apply_fitpulse_plotly_theme(fig, height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        "<div class='fitpulse-caption'>"
        "Each point represents a day. Higher steps and sufficient sleep indicate healthier behavior patterns."
        "</div>",
        unsafe_allow_html=True
        )


        st.markdown("---")

        
elif page == "Reports":

    # =========================
    # PAGE HEADER
    # =========================
    st.markdown("""
    <h2 style="
        color:#FACC15;
        text-shadow:0 0 14px rgba(250,204,21,0.7);
        margin-bottom:10px;">
        Health Reports
    </h2>
    <p style="color:#9CA3AF;margin-top:-5px;">
        View, analyze, and export your health insights
    </p>
    """, unsafe_allow_html=True)

    # =========================
    # SAFETY CHECK
    # =========================
    if (
        not st.session_state.data_loaded
        or st.session_state.health_data is None
    ):
        st.warning("⚠️ Please upload and process data first!")
        st.stop()

    # =========================
    # DATA PREPARATION
    # =========================
    df = st.session_state.health_data.copy()

    df = detect_anomalies_ml(df)

    if "ml_anomaly" not in df.columns:
        df["ml_anomaly"] = False

    ml_anomaly_count = int(df["ml_anomaly"].sum())
    health_score = calculate_health_score(df)
    risk_label, risk_color = get_risk_level(health_score, ml_anomaly_count)

    report_date = datetime.now().strftime("%d %b %Y • %I:%M %p")

    # =========================
    # REPORT OVERVIEW
    # =========================
    st.markdown("### Report Overview")

    st.markdown(f"""
    <div style="
        background:#020617;
        border:1px solid #1E293B;
        border-radius:14px;
        padding:18px;
        box-shadow:0 0 20px rgba(250,204,21,0.15);
    ">
        <h4 style="color:#FACC15;margin-bottom:8px;">
            FitPulse Health Report
        </h4>
        <p style="color:#E5E7EB;">🕒 Generated: <strong>{report_date}</strong></p>
        <p style="color:#E5E7EB;">📅 Analysis Period: <strong>{len(df)} records</strong></p>
        <p style="color:#22D3EE;">💯 Health Score: <strong>{health_score}/100</strong></p>
        <p style="color:#F87171;">⚠️ ML Anomalies Detected: <strong>{ml_anomaly_count}</strong></p>
        <p style="color:{risk_color};">🚦 Overall Risk Level: <strong>{risk_label}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # TREND COMPARISON
    # =========================
    st.markdown("### Health Trend Comparison")

    mid = max(len(df) // 2, 1)
    prev_val = df.iloc[:mid]
    curr_val = df.iloc[mid:]

    col1, col2, col3 = st.columns(3)
    def trend(curr_val, prev_val):
        if pd.isna(curr_val) or pd.isna(prev_val):
            return "➡️ 0"
        diff = curr_val - prev_val
        arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
        return f"{arrow} {abs(diff):.1f}"

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>💓 Heart Rate</h4>
                <p>{trend(curr_val['heart_rate'].mean(), 
                          prev_val['heart_rate'].mean())} BPM</p>
            </div>
        """, 
        unsafe_allow_html=True
        )

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>👟 Steps</h4>
            <p>{trend(curr_val['steps'].mean(), prev_val['steps'].mean())} steps</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>😴 Sleep</h4>
            <p>{trend(curr_val['sleep_hours'].mean(), prev_val['sleep_hours'].mean())} hrs</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # EXPORT OPTIONS
    # =========================
    st.markdown("### 📥 Export Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            " Full Data (CSV)",
            data=df.to_csv(index=False),
            file_name="health_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if ml_anomaly_count > 0:
            anomalies = df[df["ml_anomaly"]][
                ["timestamp", "heart_rate", "steps", "sleep_hours"]
            ]
            st.download_button(
                " Anomalies Only (CSV)",
                data=anomalies.to_csv(index=False),
                file_name="anomalies.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No anomalies detected")

    with col3:
        pdf_path = generate_pdf_report(
            df,
            health_score,
            ml_anomaly_count,
            risk_label
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                " FitPulse Report (PDF)",
                data=f,
                file_name="FitPulse_Health_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )



else:  # About page
    st.markdown("""
    <h2 style="color:#FACC15;text-shadow:0 0 12px rgba(250,204,21,0.6);">
    About my Application
    </h2>
    """, unsafe_allow_html=True)

    """
    **Built with ❤️ for Health & Wellness**
    """