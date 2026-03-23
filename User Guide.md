# User Guide: `mtslearn`

## 1. Data Ingestion: The Processor Base Class

The `Processor` class handles loading your raw data and converting it into a standard "wide-format" table. Both static and time-series processors build upon this base.

### `Processor.read_file`
* **What it does**: Reads CSV or Excel files and reshapes the data so that every row represents a unique combination of an `ID` and a `Timestamp`.
* **How it works**:
    * **Format Support**: It can automatically reshape `long` formats (one metric per row) or `flattened` formats (paired attribute-time columns) into a clean, wide table.
    * **Time Conversion**: It converts datetime columns into simple floating-point numbers (Unix seconds). This allows the models to easily calculate the time gaps between observations.
    * **Label Mapping**: It assigns a single target label to each unique ID (static), or dynamic labels for each time point.

---

## 2. Static Analysis: The StaticProcessor

The `StaticProcessor` is used when you want to ignore the sequential nature of the data and squash all time-steps into a single row of summary statistics for each ID (e.g., for models like XGBoost).

### `StaticProcessor.extract_features`
* **What it does**: Flattens the time-series data into static features (like mean, max, min).
* **How it works**: 
    * It calculates the requested statistics (`agg_funcs`) for every feature. 
    * It renames columns clearly (e.g., `HeartRate_mean`, `BloodPressure_max`).
    * If `include_duration=True`, it calculates how much time passed between the first and last observation and adds it as a `duration` column.
    * For `dynamic` labels, it preserves all time points but attaches global entity statistics to each row.

### `StaticProcessor.data_cleaning`
* **What it does**: Removes extreme outliers and fills in missing values (`NaN`).
* **How it works**:
    * **Leakage Prevention (Important)**: It calculates outlier thresholds (IQR or Z-score) and fill values (mean or median) **strictly on the training set**. These exact same rules are then applied to the test set.
    * Any value deemed an outlier is temporarily turned into a `NaN`, and then all `NaN` values are filled globally based on your chosen method.

### `StaticProcessor.train_test_split`
* **What it does**: Splits the flat data into training and testing sets.
* **How it works**: Randomizes the data and supports `stratify` to ensure the ratio of target classes remains balanced between the train and test sets for static labels. It also handles sequence-padding automatically if using dynamic labels.

### `StaticProcessor.scale_features` 
* **What it does**: Standardizes or normalizes the features.
* **How it works**: Uses Scikit-Learn's `StandardScaler` or `MinMaxScaler`. It fits on the training data and transforms both train and test data.

---

## 3. Time-Series Analysis: The TSProcessor

The `TSProcessor` preserves the sequence of events over time. It transforms your data into 3D tensors `(Samples, TimeSteps, Features)` required by Deep Learning models like LSTMs.

### `TSProcessor.train_test_split`
* **What it does**: Groups data by ID and builds the 3D tensors.
* **How it works**:
    * **Time Deltas**: Automatically calculates the time gap (`time_delta`) since the previous observation and adds it as a new feature.
    * **Zero-Padding & Truncation**: Finds the longest sequence (or uses a provided `max_len`). It extracts the *latest* observations up to the max length, placing them at the start of the tensor, and padding any remaining length with zeros at the end so all samples have the exact same shape.
    * Splits the data by unique ID to prevent data leakage across time steps.

### `TSProcessor.data_cleaning`
* **What it does**: Cleans 3D tensors by removing outliers and filling missing values.
* **How it works**:
    * **Smart Time Handling**: It deliberately ignores the `time_delta` column, ensuring time intervals are never accidentally altered or "cleaned".
    * **Two-Step Filling**: When filling missing values, it first looks at the specific ID's own history (e.g., carrying the last known value forward, or using that specific patient's average). If the entire history is missing, it falls back to the global average of the training set.

### `TSProcessor.scale_features` 
* **What it does**: Scales 3D sequence features safely.
* **How it works**: It applies scaling *only* to the actual data. It uses a masking technique to ensure that the zero-padding at the end of sequences is completely ignored during math calculations and remains exactly `0` after scaling.

### `TSProcessor.time_resample`
* **What it does**: Forces erratic timestamps into fixed, regular intervals (e.g., exactly every 1 hour or 1 day).
* **How it works**: Groups the data by ID, creates a new fixed time grid, and uses linear interpolation to guess what the values were at those exact fixed times.

---

## 4. Model Orchestration: The Classifier Base Classes

The Toolkit provides specific Orchestrators (`Static_Classifier` and `Temporal_Classifier`) to provide a single, unified way to train and evaluate models based on their input/output types—whether it’s a static-to-static model like XGBoost, temporal-to-static like LSTM/T-LSTM, temporal-to-temporal like Transformers, or static-to-temporal like CoxPH.

### `Classifier.fit`
* **What it does**: Trains your selected model.
* **How it works**: 
    * Merges default model settings with any custom hyperparameters you provide.
    * Triggers the model's internal training loop.
    * **Auto-Plotting**: If the model records its training loss over time, it will automatically plot a Training Loss curve for you once training finishes.

### `Classifier.evaluate`
* **What it does**: Tests the model and visually reports its performance.
* **How it works**:
    * **Text Report**: Prints a clean summary showing Precision, Recall, F1-score, and overall accuracy.
    * **Confusion Matrix & ROC Curves (Static Outputs)**: For models outputting a single label per patient, it plots a color-coded heatmap showing predictions vs truth, and automatically draws single or multi-class ROC curves.
    * **Temporal Error Distributions (Temporal Outputs)**: For models outputting a sequence of labels (e.g., Seq2Seq), it generates fine-grained histograms of sample-wise misclassification rates and time-step-wise directional errors.
