# FitPulse: Health Anomaly Detection System

A comprehensive health and fitness analytics platform that combines data preprocessing, machine learning, and advanced anomaly detection to provide actionable insights into personal health metrics.

## Overview

FitPulse is an integrated system designed to process, analyze, and detect anomalies in health and fitness data. The platform leverages modern machine learning techniques including time-series feature extraction, trend forecasting, behavioral clustering, and multi-method anomaly detection to identify health patterns and deviations.

## Features

### Core Capabilities

- **Data Preprocessing Pipeline**: Comprehensive data cleaning, validation, and time alignment for fitness tracker data
- **Feature Extraction**: Time-series feature extraction using TSFresh with statistical and entropy-based metrics
- **Trend Modeling**: Facebook Prophet integration for time-series forecasting and trend analysis
- **Behavioral Clustering**: KMeans and DBSCAN clustering for pattern identification and behavioral segmentation
- **Multi-Method Anomaly Detection**: Three complementary detection approaches for comprehensive anomaly identification
- **Interactive Visualizations**: Plotly-based interactive charts and dashboards for data exploration
- **Report Generation**: PDF and CSV export capabilities for comprehensive health reports

### Supported Data Types

- Heart rate monitoring data
- Step count and activity tracking
- Sleep duration and stage data
- Custom time-series metrics

## System Architecture

### Milestone 1: Data Preprocessing and Visualization

Handles initial data ingestion and quality assurance:

- Multi-format file support (CSV, JSON)
- Data type auto-detection
- Missing value handling and imputation
- Outlier detection and flagging
- Timestamp normalization and standardization
- Data quality reporting and validation

### Milestone 2: Feature Extraction and Trend Modeling

Implements machine learning feature engineering:

- TSFresh time-series feature extraction with 50+ metrics
- Prophet time-series forecasting with changepoint detection
- Residual-based anomaly detection from predictions
- KMeans and DBSCAN behavioral clustering
- PCA and t-SNE dimensionality reduction for visualization
- Silhouette score and Davies-Bouldin validation metrics

### Milestone 3: Advanced Anomaly Detection

Provides three complementary anomaly detection methods:

1. **Threshold-Based Detection**: Rule-based anomalies using configurable limits
2. **Residual-Based Detection**: Prophet residual analysis with statistical thresholds
3. **Cluster-Based Detection**: Outlier identification through clustering patterns

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fitpulse.git
cd fitpulse
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Launch the Streamlit application:

```bash
streamlit run main_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Application Modules

#### Dashboard Module
Main entry point providing an overview of health metrics and key statistics.

**Access**: Select "Home" from sidebar navigation

**Features**:
- Real-time health score calculation
- Average metrics display
- Anomaly count visualization
- Interactive trend charts

#### Upload Data Module
Import custom health data for analysis.

**Access**: Navigate to "Upload Data" tab

**Supported Formats**:
- CSV (comma-separated values)
- JSON (JavaScript Object Notation)

**Required Columns**:
- timestamp (or date)
- heart_rate
- steps
- sleep_hours

#### Anomaly Analysis Module
Detailed examination of detected anomalies with severity classification.

**Access**: Navigate to "Anomaly Analysis" tab

**Outputs**:
- Timeline visualization of anomalies
- Severity-based classification (high, medium, low)
- Detailed anomaly reports
- Statistical anomaly frequency analysis

#### ML Insights Module
Behavioral pattern analysis using machine learning clustering.

**Access**: Navigate to "ML Insights" tab

**Deliverables**:
- Behavioral clustering (3 distinct clusters)
- Health behavior classification
- Pattern visualization using PCA
- Personalized health recommendations

#### Reports Module
Comprehensive health report generation and export.

**Access**: Navigate to "Reports" tab

**Export Options**:
- Complete health data (CSV)
- Anomalies only (CSV)
- FitPulse health report (PDF)
- Trend comparison analysis


### Core Classes

#### FitnessDataUploader
Manages file upload and initial data loading operations.

```python
uploader = FitnessDataUploader()
data_dict = uploader.create_upload_interface()
```

#### FitnessDataValidator
Validates and cleans imported health data.

```python
validator = FitnessDataValidator()
cleaned_df, report = validator.validate_and_clean_data(df, 'heart_rate')
```

#### TimeAligner
Handles timestamp normalization and data resampling.

```python
aligner = TimeAligner()
aligned_df, report = aligner.align_and_resample(df, 'heart_rate', '1min')
```

#### TSFreshFeatureExtractor
Extracts time-series statistical features.

```python
extractor = TSFreshFeatureExtractor()
features, report = extractor.extract_features(df, 'heart_rate', window_size=60)
```

#### ProphetTrendModeler
Implements time-series forecasting and trend detection.

```python
modeler = ProphetTrendModeler()
forecast, report = modeler.fit_and_predict(df, 'heart_rate', forecast_periods=100)
```

#### BehaviorClusterer
Performs behavioral pattern clustering and analysis.

```python
clusterer = BehaviorClusterer()
labels, report = clusterer.cluster_features(features, 'heart_rate', 'kmeans', n_clusters=3)
```

#### ThresholdAnomalyDetector
Rule-based anomaly detection using threshold boundaries.

```python
detector = ThresholdAnomalyDetector()
df_result, report = detector.detect_anomalies(df, 'heart_rate')
```

#### ResidualAnomalyDetector
Model-based anomaly detection using Prophet residuals.

```python
detector = ResidualAnomalyDetector(threshold_std=3.0)
df_result, report = detector.detect_anomalies_from_prophet(df, 'heart_rate')
```

#### ClusterAnomalyDetector
Cluster-based outlier detection and identification.

```python
detector = ClusterAnomalyDetector(outlier_threshold=0.05)
df_result, report = detector.detect_cluster_outliers(features, labels, 'heart_rate')
```

## Configuration

### Default Settings

- Feature extraction window size: 60 data points
- Prophet forecast horizon: 100 periods
- Clustering method: KMeans with 3 clusters
- Anomaly threshold: 3 standard deviations
- Data validation: Strict mode enabled

### Customization

Modify configuration in the sidebar panels:

1. **Feature Window**: Adjust sliding window size for feature extraction
2. **Forecast Periods**: Set prediction horizon for trend forecasting
3. **Clustering Method**: Choose between KMeans and DBSCAN
4. **Number of Clusters**: Set cluster count for behavioral analysis
5. **Anomaly Thresholds**: Configure detection sensitivity

## Data Format Specifications

### CSV Format

```csv
timestamp,heart_rate,steps,sleep_hours
2024-01-15 08:00:00,72,0,0
2024-01-15 08:01:00,73,5,0
2024-01-15 08:02:00,71,8,0
```

### JSON Format

```json
[
  {
    "timestamp": "2024-01-15T08:00:00Z",
    "heart_rate": 72,
    "steps": 0,
    "sleep_hours": 0
  },
  {
    "timestamp": "2024-01-15T08:01:00Z",
    "heart_rate": 73,
    "steps": 5,
    "sleep_hours": 0
  }
]
```

## Output and Reports

### Health Score Calculation

The platform calculates an overall health score (0-100) based on:

- Heart rate stability relative to baseline (70 bpm)
- Daily step count relative to 10,000 step goal
- Sleep duration relative to 8-hour recommendation

### Risk Classification

- LOW: Score >= 80, no anomalies
- MEDIUM: Score >= 60, <= 3 anomalies
- HIGH: Score < 60, > 3 anomalies

### Report Contents

PDF reports include:
- Generated timestamp and analysis period
- Overall health score and risk assessment
- Average health metrics with observed ranges
- Anomaly summary and clinical notes
- Recommended actions based on detected patterns

## Performance Considerations

### Memory Usage

- Small datasets (< 10,000 records): ~100 MB
- Medium datasets (10,000 - 100,000): ~500 MB
- Large datasets (> 100,000): 1-2 GB recommended

### Processing Time

- Data validation: < 1 second per 1,000 records
- Feature extraction: 5-15 seconds per 1,000 records
- Prophet forecasting: 10-30 seconds depending on data complexity
- Clustering and anomaly detection: 2-5 seconds

## Dependencies

Core dependencies are specified in requirements.txt:

- streamlit: Web application framework
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- plotly: Interactive visualizations
- prophet: Time-series forecasting
- tsfresh: Time-series feature extraction
- reportlab: PDF generation
- scipy: Scientific computing

For complete dependency list, see requirements.txt

## Troubleshooting

### Common Issues

**Issue**: Prophet model training fails with cryptic error
- Solution: Ensure sufficient data (minimum 2 observations). Reduce forecast_periods.

**Issue**: Memory error on large datasets
- Solution: Process data in chunks or reduce window_size parameter.

**Issue**: Anomalies not detected with threshold method
- Solution: Adjust min/max threshold values in sidebar configuration.

**Issue**: Clustering produces single cluster
- Solution: Increase n_clusters parameter or check data standardization.

### Debug Mode

Enable debug information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/improvement)
3. Make your changes with clear commit messages
4. Push to the branch (git push origin feature/improvement)
5. Submit a pull request with description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2026 Arfiya Hashmi

## Support

For issues, questions, or suggestions:

1. Check existing GitHub issues
2. Review documentation
3. Create a detailed bug report or feature request
4. Contact the development team

## Roadmap

Planned features for future releases:

- Multi-user support with authentication
- Cloud data synchronization
- Mobile application integration
- Real-time alert system
- Advanced statistical testing
- Integration with popular fitness trackers
- Machine learning model fine-tuning
- Custom metric definitions

## Citation

If you use FitPulse in research or applications, please cite:

```
FitPulse: Health Anomaly Detection System
Author: Arfiya Hashmi
Year: 2026
Repository: https://github.com/yourusername/fitpulse
```

## References

- Prophet Documentation: https://facebook.github.io/prophet/
- TSFresh Documentation: https://tsfresh.readthedocs.io/
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/


**Version**: 1.0.0
**Last Updated**: January 2026
**Status**: Active Development
