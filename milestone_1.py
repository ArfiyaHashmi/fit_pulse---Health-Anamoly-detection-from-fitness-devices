import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import io
import pytz
import warnings
warnings.filterwarnings('ignore')

class FitnessDataUploader:
    """Handles file upload and initial data loading for fitness tracker data"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json']
        self.required_columns = {
            'heart_rate': ['timestamp', 'heart_rate'],
            'sleep': ['timestamp', 'sleep_stage', 'duration_minutes'],
            'steps': ['timestamp', 'step_count']
        }
    
    def create_upload_interface(self) -> Dict[str, pd.DataFrame]:
        """Create Streamlit file upload interface"""
        st.header("📁 Upload Fitness Tracker Data")
        
        uploaded_files = st.file_uploader(
            "Choose fitness data files",
            type=['csv', 'json'],
            accept_multiple_files=True,
            help="Upload CSV or JSON files containing heart rate, sleep, or step data"
        )
        
        if uploaded_files:
            data_dict = {}
            for uploaded_file in uploaded_files:
                try:
                    data_type = self._detect_data_type(uploaded_file.name)
                    
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = self._load_json_data(uploaded_file)
                    
                    if self._validate_data_structure(df, data_type):
                        data_dict[data_type] = df
                        st.success(f"✅ Successfully loaded {data_type} data: {len(df)} records")
                    else:
                        st.error(f"❌ Invalid data structure in {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
            
            return data_dict
        return {}
    
    def _detect_data_type(self, filename: str) -> str:
        filename_lower = filename.lower()
        if 'heart' in filename_lower or 'hr' in filename_lower:
            return 'heart_rate'
        elif 'sleep' in filename_lower:
            return 'sleep'
        elif 'step' in filename_lower or 'activity' in filename_lower:
            return 'steps'
        else:
            return 'unknown'
    
    def _load_json_data(self, uploaded_file) -> pd.DataFrame:
        json_data = json.load(uploaded_file)
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            if 'data' in json_data:
                df = pd.DataFrame(json_data['data'])
            else:
                df = pd.DataFrame([json_data])
        else:
            raise ValueError("Unsupported JSON structure")
        return df
    
    def _validate_data_structure(self, df: pd.DataFrame, data_type: str) -> bool:
        if data_type not in self.required_columns:
            return False
        
        required_cols = self.required_columns[data_type]
        df_columns_lower = [col.lower() for col in df.columns]
        required_cols_lower = [col.lower() for col in required_cols]
        
        for req_col in required_cols_lower:
            if req_col not in df_columns_lower:
                st.warning(f"Missing column: {req_col}")
                return False
        return True

class FitnessDataValidator:
    """Validates and cleans fitness tracker data"""
    
    def __init__(self):
        self.validation_rules = {
            'heart_rate': {'min_value': 30, 'max_value': 220, 'data_type': 'numeric'},
            'step_count': {'min_value': 0, 'max_value': 100000, 'data_type': 'numeric'},
            'duration_minutes': {'min_value': 0, 'max_value': 1440, 'data_type': 'numeric'}
        }
    
    def validate_and_clean_data(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        validation_report = {
            'original_rows': len(df),
            'issues_found': [],
            'rows_removed': 0,
            'missing_values_handled': 0,
            'outliers_flagged': 0
        }
        
        try:
            df_clean = df.copy()
            df_clean = self._standardize_columns(df_clean)
            df_clean, timestamp_issues = self._clean_timestamps(df_clean)
            validation_report['issues_found'].extend(timestamp_issues)
            
            df_clean, numeric_issues = self._validate_numeric_columns(df_clean, data_type)
            validation_report['issues_found'].extend(numeric_issues)
            
            df_clean, missing_count = self._handle_missing_values(df_clean, data_type)
            validation_report['missing_values_handled'] = missing_count
            
            df_clean, outlier_count = self._detect_outliers(df_clean, data_type)
            validation_report['outliers_flagged'] = outlier_count
            
            initial_len = len(df_clean)
            df_clean = self._remove_invalid_rows(df_clean)
            validation_report['rows_removed'] = initial_len - len(df_clean)
            validation_report['final_rows'] = len(df_clean)
            validation_report['success'] = True
            
        except Exception as e:
            validation_report['success'] = False
            validation_report['error'] = str(e)
        
        return df_clean, validation_report
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
            'hr': 'heart_rate', 'heartrate': 'heart_rate', 'heart rate': 'heart_rate',
            'steps': 'step_count', 'step': 'step_count', 'stepcount': 'step_count',
            'sleep': 'sleep_stage', 'stage': 'sleep_stage', 'duration': 'duration_minutes'
        }
        df_renamed = df.rename(columns=column_mapping)
        df_renamed.columns = df_renamed.columns.str.lower().str.replace(' ', '_')
        return df_renamed
    
    def _clean_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        issues = []
        if 'timestamp' not in df.columns:
            issues.append("No timestamp column found")
            return df, issues
        
        try:
            parsed_timestamps = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
            failed_count = parsed_timestamps.isna().sum()
            if failed_count > 0:
                issues.append(f"Failed to parse {failed_count} timestamps")
            
            df['timestamp'] = parsed_timestamps
            
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
        except Exception as e:
            issues.append(f"Timestamp processing error: {str(e)}")
        
        return df, issues
    
    def _validate_numeric_columns(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[str]]:
        issues = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in self.validation_rules:
                if col in ['step_count', 'duration_minutes']:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        issues.append(f"Found {negative_count} negative values in {col}")
                        df[col] = df[col].clip(lower=0)
        return df, issues
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, int]:
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df, 0
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col == 'timestamp':
                    df = df.dropna(subset=['timestamp'])
                elif col in ['heart_rate', 'step_count']:
                    df[col] = df[col].ffill(limit=5)
                    df[col] = df[col].interpolate(method='linear')
                elif col == 'duration_minutes':
                    median_duration = df[col].median()
                    df[col] = df[col].fillna(median_duration)
                elif col == 'sleep_stage':
                    df[col] = df[col].ffill()
        
        return df, missing_count
    
    def _detect_outliers(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, int]:
        outlier_count = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'timestamp':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                df[f'{col}_outlier'] = outliers
                outlier_count += outliers.sum()
        
        return df, outlier_count
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['timestamp'])
        value_columns = [col for col in df.columns if col not in ['timestamp'] and not col.endswith('_outlier')]
        df = df.dropna(subset=value_columns, how='all')
        return df
    
    def generate_validation_report(self, validation_report: Dict) -> str:
        report = f"""
📊 DATA VALIDATION REPORT
========================
Original rows: {validation_report['original_rows']}
Final rows: {validation_report.get('final_rows', 'N/A')}
Rows removed: {validation_report['rows_removed']}
Missing values handled: {validation_report['missing_values_handled']}
Outliers flagged: {validation_report['outliers_flagged']}

Issues Found:
"""
        if validation_report['issues_found']:
            for issue in validation_report['issues_found']:
                report += f"• {issue}\n"
        else:
            report += "• No issues found\n"
        
        return report

class TimeAligner:
    """Handles time alignment and resampling of fitness data"""
    
    def __init__(self):
        self.supported_frequencies = {
            '1min': '1T', '5min': '5T', '15min': '15T', 
            '30min': '30T', '1hour': '1H'
        }
    
    def align_and_resample(self, df: pd.DataFrame, data_type: str, 
                          target_frequency: str = '1min', fill_method: str = 'interpolate') -> Tuple[pd.DataFrame, Dict]:
        
        alignment_report = {
            'original_frequency': None, 'target_frequency': target_frequency,
            'original_rows': len(df), 'resampled_rows': 0, 'gaps_filled': 0,
            'method_used': fill_method, 'success': False
        }
        
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("No timestamp column found")
            
            df_indexed = df.set_index('timestamp')
            alignment_report['original_frequency'] = self._detect_frequency(df_indexed)
            
            if target_frequency not in self.supported_frequencies:
                raise ValueError(f"Unsupported frequency: {target_frequency}")
            
            freq_str = self.supported_frequencies[target_frequency]
            df_resampled = self._resample_by_type(df_indexed, data_type, freq_str)
            df_filled, gaps_filled = self._fill_missing_after_resample(df_resampled, data_type, fill_method)
            df_final = df_filled.reset_index()
            
            alignment_report['resampled_rows'] = len(df_final)
            alignment_report['gaps_filled'] = gaps_filled
            alignment_report['success'] = True
            
            return df_final, alignment_report
            
        except Exception as e:
            alignment_report['error'] = str(e)
            return df, alignment_report
    
    def _detect_frequency(self, df_indexed: pd.DataFrame) -> str:
        try:
            if len(df_indexed) < 2:
                return "insufficient_data"
            
            time_diffs = df_indexed.index.to_series().diff().dropna()
            mode_diff = time_diffs.mode()
            
            if len(mode_diff) == 0:
                return "irregular"
            
            mode_minutes = mode_diff.iloc[0].total_seconds() / 60
            
            if mode_minutes < 1:
                return "sub_minute"
            elif mode_minutes == 1:
                return "1min"
            elif mode_minutes == 5:
                return "5min"
            elif mode_minutes == 15:
                return "15min"
            elif mode_minutes == 30:
                return "30min"
            elif mode_minutes == 60:
                return "1hour"
            else:
                return f"{mode_minutes:.1f}min"
        except:
            return "unknown"
    
    def _resample_by_type(self, df_indexed: pd.DataFrame, data_type: str, freq_str: str) -> pd.DataFrame:
        resampled_dict = {}
        
        for column in df_indexed.columns:
            if column.endswith('_outlier'):
                resampled_dict[column] = df_indexed[column].resample(freq_str).max()
            elif column == 'heart_rate':
                resampled_dict[column] = df_indexed[column].resample(freq_str).mean()
            elif column == 'step_count':
                resampled_dict[column] = df_indexed[column].resample(freq_str).sum()
            elif column == 'duration_minutes':
                resampled_dict[column] = df_indexed[column].resample(freq_str).sum()
            elif column == 'sleep_stage':
                resampled_dict[column] = df_indexed[column].resample(freq_str).agg(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
                )
            else:
                if df_indexed[column].dtype in ['int64', 'float64']:
                    resampled_dict[column] = df_indexed[column].resample(freq_str).mean()
                else:
                    resampled_dict[column] = df_indexed[column].resample(freq_str).first()
        
        return pd.DataFrame(resampled_dict)
    
    def _fill_missing_after_resample(self, df: pd.DataFrame, data_type: str, fill_method: str) -> Tuple[pd.DataFrame, int]:
        initial_missing = df.isnull().sum().sum()
        
        if fill_method == 'interpolate':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if not col.endswith('_outlier'):
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                df[col] = df[col].ffill().bfill()
        
        elif fill_method == 'forward_fill':
            df = df.ffill()
        elif fill_method == 'backward_fill':
            df = df.bfill()
        elif fill_method == 'zero':
            df = df.fillna(0)
        elif fill_method == 'drop':
            df = df.dropna()
        
        final_missing = df.isnull().sum().sum()
        gaps_filled = initial_missing - final_missing
        return df, gaps_filled
    
    def generate_alignment_report(self, report: Dict) -> str:
        return f"""
⏰ TIME ALIGNMENT REPORT
========================
Original frequency: {report['original_frequency']}
Target frequency: {report['target_frequency']}
Original rows: {report['original_rows']}
Resampled rows: {report['resampled_rows']}
Gaps filled: {report['gaps_filled']}
Fill method: {report['method_used']}

Status: {'✅ Success' if report['success'] else '❌ Failed'}
"""

class FitnessDataPreprocessor:
    """Complete preprocessing pipeline for fitness tracker data - INTEGRATED A+B+C"""
    
    def __init__(self):
        # A. File Upload Functionality
        self.uploader = FitnessDataUploader()
        
        # B. Data Validation and Error Handling  
        self.validator = FitnessDataValidator()
        
        # C. Time Alignment and Resampling
        self.aligner = TimeAligner()
        
        self.processing_log = []
        self.processed_data = {}
        self.reports = {}
    
    def run_complete_pipeline(self, uploaded_files=None, target_frequency='1min', fill_method='interpolate') -> Dict[str, pd.DataFrame]:
        """INTEGRATED PIPELINE: A + B + C"""
        
        st.header("🔄 Data Preprocessing Pipeline")
        
        # COMPONENT A: Data Upload and Loading
        self.log_step("🔵 COMPONENT A: Starting data upload and loading...")
        
        if uploaded_files:
            raw_data = self.uploader.create_upload_interface()
        else:
            raw_data = self._create_sample_data()
        
        if not raw_data:
            st.error("No data uploaded. Please upload fitness tracker files.")
            return {}
        
        # COMPONENT B: Data Validation and Cleaning
        self.log_step("🟡 COMPONENT B: Validating and cleaning data...")
        
        validated_data = {}
        for data_type, df in raw_data.items():
            cleaned_df, validation_report = self.validator.validate_and_clean_data(df, data_type)
            validated_data[data_type] = cleaned_df
            self.reports[f"{data_type}_validation"] = validation_report
            
            st.subheader(f"📋 {data_type.title()} Validation Results")
            st.text(self.validator.generate_validation_report(validation_report))
        
        # COMPONENT C: Time Alignment and Resampling
        self.log_step("🟢 COMPONENT C: Aligning timestamps and resampling data...")
        
        aligned_data = {}
        for data_type, df in validated_data.items():
            aligned_df, alignment_report = self.aligner.align_and_resample(df, data_type, target_frequency, fill_method)
            aligned_data[data_type] = aligned_df
            self.reports[f"{data_type}_alignment"] = alignment_report
            
            st.subheader(f"⏰ {data_type.title()} Time Alignment Results")
            st.text(self.aligner.generate_alignment_report(alignment_report))
        
        # FINAL INTEGRATION
        self.log_step("✅ INTEGRATION: Final data quality checks and pipeline completion...")
        self.processed_data = aligned_data
        self._generate_processing_summary()
        
        return aligned_data
    
    def log_step(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        st.info(log_entry)
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample data demonstrating A+B+C integration"""
        
        # Generate sample heart rate data with realistic issues
        timestamps = pd.date_range(start='2024-01-15 08:00:00', end='2024-01-15 12:00:00', freq='30S')
        
        base_hr = 70
        hr_data = []
        
        for i, ts in enumerate(timestamps):
            if np.random.random() < 0.05:  # 5% missing values (tests Component B)
                hr_data.append(None)
            else:
                time_factor = np.sin(2 * np.pi * i / 120)
                noise = np.random.normal(0, 5)
                hr = base_hr + 20 * time_factor + noise
                hr_data.append(max(50, min(120, hr)))
        
        # Add some outliers for Component B testing
        hr_data[100] = 250  # Impossible heart rate
        hr_data[200] = -10  # Negative heart rate
        
        heart_rate_df = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': hr_data
        })
        
        # Generate sample step data with irregular intervals (tests Component C)
        step_timestamps = [
            '2024-01-15 08:00:00',
            '2024-01-15 08:00:30',  # 30 seconds later  
            '2024-01-15 08:02:15',  # 1:45 minutes gap
            '2024-01-15 08:05:00',  # 2:45 minutes gap
            '2024-01-15 08:05:30',  # 30 seconds later
            '2024-01-15 08:10:00',  # 4:30 minutes gap
        ]
        
        steps_df = pd.DataFrame({
            'timestamp': pd.to_datetime(step_timestamps),
            'step_count': [100, 150, 200, 250, 275, 400]
        })
        
        # Generate sample sleep data with multiple sleep stages
        sleep_timestamps = [
            '2024-01-14 22:00:00',
            '2024-01-14 22:45:00',
            '2024-01-15 00:45:00',
            '2024-01-15 02:15:00',
            '2024-01-15 03:15:00',
        ]
        
        sleep_df = pd.DataFrame({
            'timestamp': pd.to_datetime(sleep_timestamps),
            'sleep_stage': ['light_sleep', 'deep_sleep', 'rem_sleep', 'light_sleep', 'awake'],
            'duration_minutes': [45, 120, 90, 60, 5]
        })
        
        return {
            'heart_rate': heart_rate_df,
            'steps': steps_df,
            'sleep': sleep_df
        }
    
    def create_data_preview_interface(self):
        """Interactive preview showing A+B+C results"""
        
        if not self.processed_data:
            st.warning("No processed data available. Run the pipeline first.")
            return
        
        st.header("📊 Processed Data Preview (A+B+C Results)")
        
        data_type = st.selectbox("Select data type to preview:", list(self.processed_data.keys()))
        
        if data_type in self.processed_data:
            df = self.processed_data[data_type]
            
            # Component results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📁 A: Total Records", len(df))
            with col2:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("🔧 B: Data Quality", f"{100-missing_pct:.1f}%")
            with col3:
                time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                st.metric("⏰ C: Time Span", f"{time_span_hours:.1f}h")
            with col4:
                st.metric("🔗 Integration", "✅ Complete")
            
            st.subheader("Data Sample")
            st.dataframe(df.head(20), use_container_width=True)
            
            self._create_integrated_visualization(df, data_type)
    
    def _create_integrated_visualization(self, df: pd.DataFrame, data_type: str):
        """Create visualization showing A+B+C pipeline results"""
        
        st.subheader("📈 Pipeline Results Visualization")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if not col.endswith('_outlier')]
        
        if len(numeric_columns) > 0:
            selected_column = st.selectbox("Select metric:", numeric_columns, key=f"viz_{data_type}")
            
            fig = go.Figure()
            
            # Main data line (Component A: uploaded, Component C: aligned)
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df[selected_column],
                mode='lines+markers', name=selected_column.replace('_', ' ').title(),
                line=dict(width=2), marker=dict(size=4)
            ))
            
            # Highlight outliers (Component B: validation)
            outlier_col = f"{selected_column}_outlier"
            if outlier_col in df.columns:
                outlier_data = df[df[outlier_col] == True]
                if not outlier_data.empty:
                    fig.add_trace(go.Scatter(
                        x=outlier_data['timestamp'], y=outlier_data[selected_column],
                        mode='markers', name='B: Outliers Detected',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
            
            fig.update_layout(
                title=f"A+B+C Pipeline: {selected_column.replace('_', ' ').title()} Processing Results",
                xaxis_title="C: Aligned Timestamps", yaxis_title="A: Uploaded Values (B: Validated)",
                hovermode='x unified', height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _generate_processing_summary(self):
        """Generate comprehensive A+B+C processing summary"""
        
        st.header("📝 Complete Pipeline Summary (A+B+C)")
        
        # Show integration of all components
        st.subheader("🔄 Pipeline Component Integration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("🔵 **Component A: File Upload**")
            st.write("✅ Multi-format file support")
            st.write("✅ Data type auto-detection") 
            st.write("✅ Structure validation")
        
        with col2:
            st.success("🟡 **Component B: Data Validation**")
            st.write("✅ Missing value handling")
            st.write("✅ Outlier detection")
            st.write("✅ Data quality checks")
        
        with col3:
            st.success("🟢 **Component C: Time Alignment**")
            st.write("✅ Timestamp normalization")
            st.write("✅ Frequency resampling")
            st.write("✅ Gap filling")
        
        # Processing log
        st.subheader("Processing Log")
        for log_entry in self.processing_log:
            st.text(log_entry)
        
        # Summary statistics
        summary_data = []
        for data_type, df in self.processed_data.items():
            summary_data.append({
                'Data Type': data_type.replace('_', ' ').title(),
                'A: Records Loaded': len(df),
                'B: Quality Score': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%",
                'C: Aligned Duration': f"{(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f}h",
                'Pipeline Status': "✅ A+B+C Complete"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.subheader("A+B+C Integration Summary")
        st.dataframe(summary_df, use_container_width=True)

def main():
    st.set_page_config(page_title="Fitness Data Preprocessor", page_icon="🏃‍♂️", layout="wide")
    
    st.title("🏃‍♂️ Fitness Data Preprocessing Pipeline")
    st.markdown("**Milestone 1: Complete A+B+C Integration**")
    
    # Highlight the integration
    st.info("🔗 **Integrated Components:** A) File Upload + B) Data Validation + C) Time Alignment")
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = FitnessDataPreprocessor()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Fitness Tracker")
    
    target_frequency = st.sidebar.selectbox(
        "C: Target Frequency:", options=['1min', '5min', '15min', '30min', '1hour'], index=0
    )
    
    fill_method = st.sidebar.selectbox(
        "C: Missing Value Fill Method:", 
        options=['interpolate', 'forward_fill', 'backward_fill', 'zero', 'drop'], index=0
    )
    
    use_sample_data = st.sidebar.checkbox("Use Sample Data (demonstrates A+B+C)", value=True)
    
    # Main processing
    if st.button("🚀 Run Complete A+B+C Pipeline", type="primary"):
        with st.spinner("Processing through A+B+C pipeline..."):
            processed_data = st.session_state.preprocessor.run_complete_pipeline(
                uploaded_files=None if use_sample_data else "upload",
                target_frequency=target_frequency, fill_method=fill_method
            )
            
            if processed_data:
                st.success("✅ Complete A+B+C data preprocessing pipeline completed successfully!")
            else:
                st.error("❌ A+B+C pipeline failed.")
    
    # Data preview interface
    st.session_state.preprocessor.create_data_preview_interface()

if __name__ == "__main__":
    main()

