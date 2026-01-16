# User Guide: `mtslearn`

## 1. Data Ingestion: The Processor Base Class

The `Processor` class is an abstract base class designed to define the standard protocol for data loading and format conversion. It is not intended to be instantiated directly but provides the core `read_file` logic for its subclasses, `StaticProcessor` and `TSProcessor`.

### `Processor.read_file`

**1. Functionality**
This function loads raw data from CSV or Excel files and transforms it into a structured "wide-format" DataFrame, where rows are indexed by a combination of `ID` and `Timestamp`.

**2. Operational Details**

* **Inheritance Role**: This method establishes a unified data structure, ensuring that both static and temporal subclasses operate on the same underlying schema.
* **Format Transformation**:
    * **Long Format**: Uses `pivot_table` to expand multiple rows of measurements into columns based on `id_col` and `time_col`.
    * **Flattened Format**: Parses pairs of `(Attribute, Time)` columns, stacks them, and reshapes them into a feature-oriented wide table.


* **Handling of Invariant Variables**: Variables that do not change over time (e.g., gender, race) are treated as constant time-series variables, with their values replicated across all timestamps for that ID.
* **Timestamp Normalization**: All time data is converted into Unix Epoch seconds (float), enabling numerical calculations for time intervals in downstream modules.
* **Label Alignment**: If a `label_col` is provided, the function performs a `groupby(id_col)` and selects the first available label to serve as the ground truth for that specific entity.

---

## 2. Static Analysis: The StaticProcessor

The `StaticProcessor` inherits from `Processor`. Its primary purpose is to collapse the temporal dimension of sequence data into a single-dimensional static feature vector per ID.

### `StaticProcessor.extract_features`

**1. Functionality**
This function aggregates all time-series observations for each ID into summary statistics, effectively removing the time dimension.

**2. Operational Details**

* **Statistical Expansion**: For every feature in the input list, the function applies each operation defined in `agg_funcs` (e.g., `mean`, `std`, `max`). This results in a feature space expansion where the total number of columns equals `len(features) * len(agg_funcs)`.
* **Temporal Duration**: If `include_duration=True`, it calculates the difference between the final and first timestamps for each ID, appending a new feature named `duration`.
* **Explicit Naming**: To maintain independence and traceability, new columns are named following the strict `FeatureName_FunctionName` convention.

### `StaticProcessor.data_cleaning`

**1. Functionality**
This function performs outlier detection and missing value imputation within the aggregated static feature space.

**2. Operational Details**

* **Independent Column Analysis**: Statistical thresholds are calculated independently for each aggregated column.
* **IQR Mode**: Values outside the range  are replaced with `NaN`.
* **Z-Score Mode**: Values deviating from the mean by more than 3 standard deviations are replaced with `NaN`.


* **Global Imputation**: Once outliers are removed, `NaN` values are filled using the global mean or median of that specific column derived from the entire training population.

### `StaticProcessor.train_test_split`

**1. Functionality**
Partitions the static feature matrix into training and testing sets, converting the data into 2D NumPy arrays.

**2. Operational Details**

* **Standardization Isolation**: When `standardize=True`, the `StandardScaler` is fitted exclusively on the training set. This scaler is then used to transform the test set, preventing information leakage from future data.
* **Stratified Sampling**: Supports the `stratify` parameter to ensure that the class distribution in the splits remains consistent with the original dataset proportions.

---

## 3. Time-Series Analysis: The TSProcessor

The `TSProcessor` inherits from `Processor`. It maintains the temporal structure of the data and is responsible for generating the 3D tensors required for deep learning models.

### `TSProcessor.data_cleaning`

**1. Functionality**
This function cleans individual observations within a sequence while preserving the integrity of the temporal axis.

**2. Operational Details**

* **Timestamp Protection**: During the cleaning process, the timestamp column is explicitly excluded from outlier detection to prevent the corruption of the temporal sequence.
* **Cascaded Imputation**:
1. **Local Forward-Fill**: It first attempts to fill `NaN` values using the most recent previous observation for that specific ID (`ffill`).
2. **Global Back-Fill**: Any remaining `NaN`s (typically at the start of a sequence) are filled using the global mean of that feature across all IDs.


* **Value Clamping**: In addition to standard outlier methods, it provides a `clamp` mode that clips all values to the 5th and 95th percentile range.

### `TSProcessor.time_resample`

**1. Functionality**
Standardizes the observation frequency for each ID, creating equidistant intervals between data points.

**2. Operational Details**

* **Frequency Regularization**: Generates a new time grid based on the `freq` parameter (e.g., '1H' for hourly).
* **Linear Interpolation**: Values for the newly created time steps are calculated using linear interpolation between existing points.
* **Sequence Length Maintenance**: Resampling ensures a consistent time delta within an ID, but it does not force different IDs to have the same total number of observations.

### `TSProcessor.train_test_split`

**1. Functionality**
Transforms the wide-format DataFrame into a 3D NumPy tensor of shape `(Samples, TimeSteps, Features)`.

**2. Operational Details**

* **ID-Based Partitioning**: The split logic operates on the unique ID list. This ensures that all time steps belonging to a single entity are kept together in either the training or testing set.
* **Temporal Delta Calculation**: The function automatically computes a `time_delta` feature (the interval since the previous observation) and appends it as the final feature channel in the tensor.
* **Automatic Zero-Padding**: The system identifies the maximum sequence length (`max_steps`) in the dataset. Sequences shorter than this length are padded with zeros at the end of the sequence to ensure uniform tensor dimensions.

---

## 4. Model Orchestration: The Classifier Base Class

`Classifier` is a base class providing a unified interface for model training and evaluation. It is inherited by `StaticClassifier` (for tabular models) and `TSClassifier` (for sequence models).

### `Classifier.evaluate`

**1. Functionality**
Generates a comprehensive performance report and visualization suite for a trained model.

**2. Operational Details**

* **Metrics Generation**: Produces a standard classification report containing Precision, Recall, and F1-score.
* **Confusion Matrix Visualization**: Renders a heatmap comparing predicted labels against true labels.
* **ROC Strategy**:
* For binary tasks, it plots the standard ROC curve.
* For multi-class tasks, it employs a "One-vs-Rest" (OvR) approach, calculating an independent ROC curve and AUC score for every individual class.



### `StaticClassifier.fit` (Inherits from Classifier)

**1. Functionality**
Initializes and trains non-temporal models, such as XGBoost.

**2. Operational Details**

* **Integrated Monitoring**: Automatically assigns the training set as the `eval_set` during the fit process. This allows the model to capture and store the loss history for every boosting round.
* **Configuration Merging**: Merges user-defined `model_config` parameters with internal `DEFAULT_CONFIGS` to instantiate the underlying model.

### `TSClassifier.fit` (Inherits from Classifier)

**1. Functionality**
Initializes and trains deep learning sequence models like LSTM or Time-Aware LSTM (T-LSTM).

**2. Operational Details**

* **Dynamic Tensor Adaptation**: The input layer size of the neural network is automatically determined by the third dimension (feature count) of the input training tensor.
* **Time-Aware Processing**: For `T-LSTM` models, the class uses the `time_index` to isolate the time-delta column. This column is fed into the specialized time-decay gates of the T-LSTM cell, while all other features are processed through standard LSTM gates.
* **Convergence Visualization**: Upon completion of training, it utilizes the `loss_history` attribute to automatically trigger the base class plotting functions to display the training curve.