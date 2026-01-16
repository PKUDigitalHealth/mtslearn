# Medical Time-Series Analysis Toolkit (`mtslearn`)

The **Medical Time-Series Analysis Toolkit** `mtslearn` is designed to process and analyze complex, irregularly sampled medical data. It provides a streamlined pipeline from raw data cleaning and resampling to advanced predictive modeling using both Machine Learning (XGBoost) and Deep Learning (T-LSTM).

## 🌟 Key Features

* **Multi-Format Data Support**: Handles both **Static** and **Time-Series** data processing workflows.
* **Flexible Data Ingestion**: Supports both **Wide** and **Long** data formats commonly found in clinical electronic health records (EHR).
* **Advanced Temporal Modeling**: Features **T-LSTM (Time-Aware LSTM)** to specifically handle irregular time intervals between patient visits.
* **Diverse Model Integration**: Supports a variety of architectures, from XGBoost for static features to LSTM and T-LSTM for time series features .
* **End-to-End Pipeline**: Integrated modules for data cleaning, outlier detection, resampling, standardization, and performance evaluation (ROC/Confusion Matrix).

## 🛠 Installation

You can now install the toolkit and all its dependencies directly via pip:

```bash
pip install mtslearn

```

## 🚀 Quick Start
Get started in seconds by running our interactive tutorial: 🔗 [test.ipynb](test.ipynb): This notebook demonstrates the complete workflow of static and time series processes using a sample dataset containing 375 patients.

## Documentation

For detailed documentation, including advanced usage, customization options, and examples, refer to the [User Guide](User%20Guide.md) .

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact us as 202363010015@nuist.edu.cn.
