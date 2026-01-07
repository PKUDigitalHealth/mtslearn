# Medical Time-Series Analysis Toolkit (`mtslearn`)

The **Medical Time-Series Analysis Toolkit** `mtslearn` is designed to process and analyze complex, irregularly sampled medical data. It provides a streamlined pipeline from raw data cleaning and resampling to advanced predictive modeling using both Machine Learning (XGBoost) and Deep Learning (T-LSTM).

## 🌟 Key Features

* **Multi-Format Data Support**: Handles both **Static** and **Time-Series** data processing workflows.
* **Flexible Data Ingestion**: Supports both **Wide** and **Long** data formats commonly found in clinical electronic health records (EHR).
* **Advanced Temporal Modeling**: Features **T-LSTM (Time-Aware LSTM)** to specifically handle irregular time intervals between patient visits.
* **Diverse Model Integration**: Supports a variety of architectures, from XGBoost for static features to LSTM and T-LSTM for time series features .
* **End-to-End Pipeline**: Integrated modules for data cleaning, outlier detection, resampling, standardization, and performance evaluation (ROC/Confusion Matrix).

---

## 🛠 Installation

You can now install the toolkit and all its dependencies directly via pip:

```bash
pip install mtslearn

```

---

## 🚀 Quickstart

### 1. Configuration & Data Formats

Define your data structure using the config dictionary. mtslearn supports two primary formats to accommodate different clinical data export styles:
#### **Wide Format**

Each row represents a single time point with all measurements in separate columns.
| PATIENT_ID | RE_DATE | Heart Rate | Steps | outcome |
| :--- | :--- | :--- | :--- | :--- |
| U001 | 08:00 | 72 | 100 | 0 |
| U001 | 08:01 | 75 | 120 | 0 |
| U002 | 08:00 | 68 | 50 | 1 |

#### **Long Format**

Each row represents a single measurement entry, identifying the attribute name and its value.
| PATIENT_ID | RE_DATE | attribute | value | outcome |
| :--- | :--- | :--- | :--- | :--- |
| U001 | 08:00 | Heart Rate | 72 | 0 |
| U001 | 08:00 | Steps | 100 | 0 |
| U001 | 08:01 | Heart Rate | 75 | 0 |
| U002 | 08:00 | Heart Rate | 68 | 1 |

```python
from mtslearn import TSClassifier, TSProcessor, StaticProcessor, StaticClassifier

config = {
    "file_path": r'test_data/375_patients_example.xlsx',  # path to data
    "data_type": 'wide',    # Define format: 'long' or 'wide'
    "num_type": 2,          # numbers of classes
    "time_col": 'RE_DATE',  # Specify time column
    "id_col": 'PATIENT_ID', # Specify grouping column
    "label_col": 'outcome', # Specify label column
    "value_col": 'value',   # Required for Long format
    "attr_col": 'attribute' # Required for Long format
}

```

---

### 2. Static Data Pipeline (Machine Learning)

Use `StaticProcessor` to aggregate time-series records into static clinical features for models like XGBoost.

```python
# Initialize and read
static_processor = StaticProcessor()
static_processor.read_file(**config)

# Feature extraction (Mean, Std, etc.)
agg_params = {
    'agg_funcs': ['mean', 'std', 'max', 'min', 'median'], 
    'include_duration': True
}
static_processor.extract_features(**agg_params)

# Data cleaning
static_processor.data_cleaning(fill_missing='mean', outlier_method='iqr')

# Train/Test Split
X_train, X_test, y_train, y_test = static_processor.train_test_split(
    test_size=0.2, 
    standardize=True
)

# Classification with XGBoost
model = StaticClassifier(model_type='XGB')
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)

```

---

### 3. Time-Series Pipeline (Deep Learning)

Use `TSProcessor` to maintain the temporal sequence and handle irregular sampling via resampling.

```python
# Initialize and clean
processor = TSProcessor()
processor.read_file(**config)
processor.data_cleaning(fill_missing='mean', outlier_method='iqr')

# Resample to align irregular time steps
processor.time_resample(freq='1D', fill_method='linear')  

# Prepare sequential data
X_train, X_test, y_train, y_test = processor.train_test_split(
    test_size=0.2, 
    standardize=True
)

# Training T-LSTM to capture longitudinal dependencies
model = TSClassifier(
    model_type='T-LSTM', 
    hidden_size=64,
    time_index=processor.time_index  
)
model.fit(X_train, y_train, epochs=50, lr=0.005)
model.evaluate(X_test, y_test)

```

For a more complete workflow demonstration, please refer to our user demo notebook: [test.ipynb](test.ipynb): This notebook demonstrates the complete workflow of static and time series processes using a sample dataset containing 375 patients.

## **Documentation**

For detailed documentation, including advanced usage, customization options, and examples, refer to the [User Guide](User%20Guide.md) .

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact us as 202363010015@nuist.edu.cn.
