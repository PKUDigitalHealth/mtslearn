# Medical Time-Series Analysis Toolkit (`mtslearn`)

The **Medical Time-Series Analysis Toolkit** `mtslearn` is designed to process and analyze complex, irregularly sampled medical data. It provides a streamlined pipeline from raw data cleaning and resampling to advanced predictive modeling using both Machine Learning (XGBoost) and Deep Learning (T-LSTM).

## 🌟 Key Features

* **Multi-Format Data Support**: Handles both **Static** and **Time-Series** data processing workflows.
* **Flexible Data Ingestion**: Supports both **Wide** and **Long** data formats commonly found in clinical electronic health records (EHR).
* **Advanced Temporal Modeling**: Features **T-LSTM (Time-Aware LSTM)** to specifically handle irregular time intervals between patient visits.
* **Diverse Model Integration**: Supports a variety of architectures, from XGBoost and CoxPH for static features to LSTM, T-LSTM, and Transformer for time series features.
* **End-to-End Pipeline**: Integrated modules for data cleaning, outlier detection, resampling, standardization, and performance evaluation (ROC/Confusion Matrix for static outputs, Error Distributions for temporal outputs).

## 🛠 Installation

You can now install the toolkit and all its dependencies directly via pip:

```bash
pip install mtslearn

```

## 🚀 Quick Start

1. Data Loading & Feature Engineering
```python
from mtslearn import StaticProcessor, Static_Static_Classifier

static_processor = StaticProcessor()
static_processor.load_dataset("COVID-19")  # Built-in dataset
static_processor.extract_features(agg_funcs=['mean', 'std', 'max', 'min', 'median'], include_duration=True)
```

2. Data Preprocessing & Cleaning
```python
X_train_static, X_test_static, y_train_static, y_test_static = static_processor.train_test_split(
    test_size=0.3, shuffle=True, random_state=42, stratify=True
) # data Splitting
X_train_static, X_test_static = static_processor.data_cleaning(
    X_train_static, X_test_static, fill_missing='mean', outlier_method='iqr'
) # data cleaning
# standardization
X_train_static, X_test_static = static_processor.scale_features(X_train_static, X_test_static, method='standardize')
```

3. Model Training & Evaluation
```python
model = Static_Static_Classifier(model_type='XGB') 
model.fit(X_train_static, y_train_static)
model.evaluate(X_test_static, y_test_static)
```

For more in-depth examples, refer to 🔗 [test.ipynb](test.ipynb), which demonstrates the complete workflow for both static and time-series processes.

## Documentation

For detailed documentation, including advanced usage, customization options, and examples, refer to the [User Guide](User%20Guide.md) .

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact us as 202363010015@nuist.edu.cn.
