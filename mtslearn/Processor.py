import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings


class Processor:

    def __init__(self):
        self.wide_df = None
        self.id_col = None
        self.time_col = None
        self.label_col = None
        self.features = []

    def read_file(self, file_path, time_col, id_col, label_col=None, value_col=None, attr_col=None, data_type='long'):
        """
        Read data from Excel and convert long-format data to wide-format.

        Parameters:
        - file_path: Path to the Excel file.
        - time_col: Name of the timestamp column.
        - id_col: Name of the ID column.
        - label_col: Name of the target label column.
        - value_col: Name of the metric value column (for long format).
        - attr_col: Name of the attribute/feature name column (for long format).
        - data_type: Format of input data ('long' or 'wide').
        """
        self.id_col = id_col
        self.time_col = time_col
        self.label_col = label_col

        if file_path.endswith('.csv') or file_path.endswith('.txt'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if data_type == 'long':
            # Transform multiple rows of metrics into a single row per ID and timestamp
            pivot_idx = [id_col, time_col]
            if label_col:
                pivot_idx.append(label_col)
            self.wide_df = df.pivot_table(index=pivot_idx, columns=attr_col, values=value_col).reset_index()

        elif data_type == 'flattened':
            # 1. Determine ID and Label
            self.id_col = df.columns[0]
            self.label_col = df.columns[-1]

            # 2. Obtain intermediate observation pairs (Attr_n, Time_n)
            obs_cols = df.columns[1:-1]

            # 3. Flattened data
            rows = []
            for _, row in df.iterrows():
                for i in range(0, len(obs_cols), 2):
                    # Get feature names (from column names, such as "Attr 1"）
                    attr_name = obs_cols[i]
                    attr_val = row[obs_cols[i]]
                    time_val = row[obs_cols[i + 1]]

                    if pd.notna(attr_val) and pd.notna(time_val):
                        rows.append({
                            self.id_col: row[self.id_col],
                            self.label_col: row[self.label_col],
                            self.time_col: time_val,
                            "__attr_name__": attr_name,
                            "__value__": attr_val
                        })

            # 4. Core transformation: Expanding different Attrs into independent feature columns using pivot.
            temp_df = pd.DataFrame(rows)
            temp_df[self.time_col] = pd.to_datetime(temp_df[self.time_col])

            # Convert to Wide Format: ID + Time + Label + Various Attributes
            self.wide_df = temp_df.pivot_table(
                index=[self.id_col, self.time_col, self.label_col], columns="__attr_name__", values="__value__"
            ).reset_index()

            # Clean up column name hierarchy
            self.wide_df.columns.name = None
        else:
            self.wide_df = df
        self.wide_df[time_col] = pd.to_datetime(self.wide_df[time_col])

        # Identify feature columns by excluding metadata columns
        exclude = [self.id_col, self.time_col, self.label_col]
        self.features = [c for c in self.wide_df.columns if c not in exclude]

        # Convert time information into floating-point numbers
        date_cols = self.wide_df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if col != self.time_col:
                self.wide_df[col] = self.wide_df[col].astype(np.int64) // 10**9

        if self.label_col:
            # Map a single ground truth label to each unique ID
            self.y = self.wide_df.groupby(self.id_col)[self.label_col].first()

    def export(self, export_path):
        """
        Save the processed wide-format DataFrame to Excel.

        Parameters:
        - export_path: Path where the output Excel file will be saved.
        """
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        self.wide_df.to_excel(export_path, index=False)


class TSProcessor(Processor):

    def data_cleaning(self, X_train, X_test, fill_missing='mean', outlier_method='iqr'):
        """
        Perform statistical cleaning on 3D Tensor features.
        Operates on (Samples, TimeSteps, Features) tensors.

        Parameters:
        - X_train (np.ndarray): 3D Tensor [Samples, TimeSteps, Features].
        - X_test (np.ndarray): 3D Tensor [Samples, TimeSteps, Features].
        - fill_missing (str): 'mean', 'median', 'ffill' (intra-sample first, then global train).
        - outlier_method (str): 'iqr', 'zscore', 'clamp'.

        Returns:
        - X_train, X_test (np.ndarray): Cleaned 3D Tensors.
        """
        # Ensure inputs are float type to allow np.nan assignment
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        n_features = X_train.shape[2]

        # Iterate over each feature channel
        for i in range(n_features):
            # Skip the time_delta column (usually we don't clean the time interval feature)
            if hasattr(self, 'time_index') and i == self.time_index:
                continue

            # --- 1. Outlier Detection (Based strictly on X_train stats) ---

            # Flatten X_train to calculate global statistics for this feature
            # Note: We use nan-safe functions to ignore any existing NaNs
            train_flat = X_train[:, :, i].flatten()
            train_valid = train_flat

            if outlier_method == 'iqr':
                q1 = np.nanquantile(train_valid, 0.25)
                q3 = np.nanquantile(train_valid, 0.75)
                iqr = q3 - q1
                lower = q1 - 1.5*iqr
                upper = q3 + 1.5*iqr

                # Apply mask (Set outliers to NaN)
                X_train[(X_train[:, :, i] < lower) | (X_train[:, :, i] > upper), i] = np.nan
                X_test[(X_test[:, :, i] < lower) | (X_test[:, :, i] > upper), i] = np.nan

            elif outlier_method == 'zscore':
                mean = np.nanmean(train_valid)
                std = np.nanstd(train_valid)
                if std > 0:
                    # Apply z-score threshold
                    X_train[np.abs(X_train[:, :, i] - mean) > 3 * std, i] = np.nan
                    X_test[np.abs(X_test[:, :, i] - mean) > 3 * std, i] = np.nan

            elif outlier_method == 'clamp':
                low = np.nanquantile(train_valid, 0.05)
                high = np.nanquantile(train_valid, 0.95)
                # Clip values
                X_train[:, :, i] = np.clip(X_train[:, :, i], low, high)
                X_test[:, :, i] = np.clip(X_test[:, :, i], low, high)

            # --- 2. Missing Value Imputation (Sample-level first, then Global Train) ---

            # Calculate Global Statistic from Train (Fallback)
            if fill_missing == 'median':
                global_fill = np.nanmedian(train_valid)
            elif isinstance(fill_missing, (int, float)):
                global_fill = fill_missing
            else:  # default to mean
                global_fill = np.nanmean(train_valid)

            # If global stat is NaN (empty column), fill with 0
            if np.isnan(global_fill):
                global_fill = 0.0

            # Define helper to apply filling per sample (ID)
            def fill_channel(tensor_channel, strategy):
                # tensor_channel shape: (Samples, TimeSteps)

                if strategy == 'ffill':
                    # Use pandas ffill purely for convenience on rows
                    # Convert to DataFrame: Rows=Samples, Cols=TimeSteps
                    # DataFrame.ffill(axis=1) fills along TimeSteps (columns)
                    df_temp = pd.DataFrame(tensor_channel)
                    df_temp = df_temp.ffill(axis=1)
                    return df_temp.values

                elif strategy == 'mean':
                    # Calculate mean per sample (Row-wise mean), ignoring NaNs
                    # shape: (Samples, 1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        sample_means = np.nanmean(tensor_channel, axis=1, keepdims=True)
                    # Replace NaNs in the tensor with the corresponding sample mean
                    # np.where(condition, x, y)
                    return np.where(np.isnan(tensor_channel), sample_means, tensor_channel)

                elif strategy == 'median':
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        sample_medians = np.nanmedian(tensor_channel, axis=1, keepdims=True)
                    return np.where(np.isnan(tensor_channel), sample_medians, tensor_channel)

                return tensor_channel

            # Apply Imputation
            if isinstance(fill_missing, str):
                # 1. Intra-Sample Fill (ID specific)
                # Fill X_train using its own sample history/stats
                X_train[:, :, i] = fill_channel(X_train[:, :, i], fill_missing)
                # Fill X_test using its own sample history/stats (Not leakage: using own past/context)
                X_test[:, :, i] = fill_channel(X_test[:, :, i], fill_missing)

                # 2. Global Fallback (using Train stats)
                # Fill any remaining NaNs (e.g., start of sequence for ffill, or all-NaN samples)
                mask_train = np.isnan(X_train[:, :, i])
                mask_test = np.isnan(X_test[:, :, i])

                X_train[mask_train, i] = global_fill
                X_test[mask_test, i] = global_fill

            else:
                # Constant fill
                mask_train = np.isnan(X_train[:, :, i])
                X_train[mask_train, i] = global_fill
                mask_test = np.isnan(X_test[:, :, i])
                X_test[mask_test, i] = global_fill

        return X_train, X_test

    def time_resample(self, freq='1D', fill_method='linear'):
        """
        Resample time series to a fixed frequency per ID. 
        Maintains wide-format DataFrame. Sequence lengths may still vary between IDs.

        Parameters:
        - freq: Resampling frequency (e.g., '1D', 'H').
        - fill_method: Interpolation method for missing time steps.
        """
        if self.wide_df[[self.id_col, self.time_col]].isnull().any().any():
            raise ValueError("ID or Time column contains NaN; resampling aborted")

        resampled_groups = []
        unique_ids = self.wide_df[self.id_col].unique()

        for _id in unique_ids:
            # Filter and set time as index for resampling
            group = self.wide_df[self.wide_df[self.id_col] == _id].set_index(self.time_col)

            # Resample features and interpolate
            resampled = group[self.features].resample(freq).mean()
            resampled = resampled.interpolate(method=fill_method).ffill().bfill()

            # Restore ID column and reset index to keep time_col as a column
            resampled.insert(0, self.id_col, _id)
            resampled_groups.append(resampled.reset_index())

        self.wide_df = pd.concat(resampled_groups, ignore_index=True)
        return self.wide_df

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None, standardize=True):
        """
        Splits data into 3D tensors by unique IDs for sequence modeling.
        
        Parameters:
        - test_size (float): Fraction of IDs for testing.
        - shuffle (bool): Whether to randomize ID order.
        - random_state (int): Seed for reproducibility.
        - standardize (bool): Scale features based on training set statistics.

        Returns:
        - X_train, X_test (np.ndarray): 3D Tensors [Samples, TimeSteps, Features].
        - y_train, y_test (np.ndarray): Corresponding labels.
        """
        # --- 1. Feature Engineering & Sorting ---
        # Sort by ID and Time to ensure time_delta calculation is chronologically correct
        df_sorted = self.wide_df.sort_values([self.id_col, self.time_col])
        # Diff within groups to prevent time gaps leaking between different IDs
        df_sorted['time_delta'] = df_sorted.groupby(self.id_col)[self.time_col].diff().dt.total_seconds().fillna(0)

        current_features = self.features + ['time_delta']
        self.time_index = len(current_features) - 1

        # --- 2. ID-based Split ---
        unique_ids = df_sorted[self.id_col].unique()
        # Slice indices based on the calculated split ratio
        if isinstance(test_size, float):
            n_test = int(len(unique_ids) * test_size)
        else:
            n_test = test_size

        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(unique_ids)

        test_ids = unique_ids[-n_test:]
        train_ids = unique_ids[:-n_test]

        # --- 3. Standardization (Train-fit, Global-transform) ---
        if standardize:
            scaler = StandardScaler()
            # Fit only on training set to avoid data leakage (look-ahead bias)
            train_mask = df_sorted[self.id_col].isin(train_ids)
            scaler.fit(df_sorted.loc[train_mask, current_features])
            df_sorted[current_features] = scaler.transform(df_sorted[current_features])
            self.scaler = scaler

        # --- 4. 3D Tensor Construction ---
        max_steps = df_sorted.groupby(self.id_col).size().max()
        n_features = len(current_features)

        def build_tensor(id_list):
            tensor = np.zeros((len(id_list), max_steps, n_features))
            # Dictionary mapping avoids repeated O(N) filtering in the loop
            grouped = dict(list(df_sorted.groupby(self.id_col)))
            for i, _id in enumerate(id_list):
                group_data = grouped[_id][current_features].values
                # Zero-padding for sequences shorter than max_steps
                tensor[i, :len(group_data), :] = group_data
            return tensor

        X_train = build_tensor(train_ids)
        X_test = build_tensor(test_ids)

        # --- 5. Label Alignment ---
        # Ensure label order strictly matches the shuffled X tensor order
        y_train = self.y.loc[train_ids].values
        y_test = self.y.loc[test_ids].values

        return X_train, X_test, y_train, y_test


class StaticProcessor(Processor):

    def extract_features(self, agg_funcs=['mean', 'std', 'max', 'min'], include_duration=False):
        """
        Aggregate time-series data into static statistical features for each unique ID.
        Flattens temporal variance into a single row per instance.

        Parameters:
        - agg_funcs (list): List of aggregation strings (e.g., 'mean', 'std', 'max', 'min', 'skew').
        - include_duration (bool): Whether to include the time duration (max - min) as a feature.

        Returns:
        - wide_df (pd.DataFrame): Aggregated DataFrame with flattened feature columns.
        """
        # Group temporal observations by ID and apply multi-statistic aggregation
        agg_df = self.wide_df.groupby(self.id_col)[self.features].agg(agg_funcs)

        # Flatten MultiIndex columns into 'Feature_Function' naming convention
        agg_df.columns = [f"{col}_{func}" for col, func in agg_df.columns]

        # Calculate duration if requested (Difference between max and min timestamp)
        if include_duration:
            duration = self.wide_df.groupby(self.id_col)[self.time_col].agg(lambda x: x.max() - x.min())
            agg_df['duration'] = duration.dt.total_seconds()

        self.features = list(agg_df.columns)

        # Synchronize labels by taking the first observation per ID group
        if self.label_col:
            labels = self.wide_df.groupby(self.id_col)[self.label_col].first()
            self.wide_df = pd.concat([agg_df, labels], axis=1).reset_index()
            self.y = self.wide_df[self.label_col]
        else:
            self.wide_df = agg_df.reset_index()

        return self.wide_df

    def data_cleaning(self, X_train, X_test, fill_missing='mean', outlier_method='iqr'):
        """
        Perform statistical cleaning on aggregated features. Detects outliers via IQR/Z-Score based on X_train statistics and imputes missing values.

        Parameters:
        - X_train (np.ndarray or pd.DataFrame): The training feature set.
        - X_test (np.ndarray or pd.DataFrame): The testing feature set.
        - fill_missing (str): Imputation method ('mean', 'median') calculated on X_train.
        - outlier_method (str): Logic for detection ('iqr' or 'zscore') derived from X_train.

        Returns:
        - X_train (np.ndarray): The cleaned training set.
        - X_test (np.ndarray): The cleaned testing set.
        """
        # Ensure inputs are numpy arrays and float type to handle np.nan
        X_train = X_train.values if hasattr(X_train, 'values') else X_train
        X_test = X_test.values if hasattr(X_test, 'values') else X_test

        # Create copies and ensure float type (integers cannot hold np.nan)
        X_train = X_train.astype(float).copy()
        X_test = X_test.astype(float).copy()

        n_features = X_train.shape[1]

        # Iterate through features to flag and nullify extreme statistical outliers
        # CRITICAL: Statistics must be calculated ONLY on X_train to avoid data leakage
        for i in range(n_features):
            train_col = X_train[:, i]

            if outlier_method == 'iqr':
                # Use Interquartile Range from TRAIN set (using nanquantile to ignore existing NaNs)
                q1 = np.nanquantile(train_col, 0.25)
                q3 = np.nanquantile(train_col, 0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5*iqr
                upper_bound = q3 + 1.5*iqr

                # Apply outlier masking (set to NaN)
                # Note: We use boolean indexing directly on the array
                X_train[(X_train[:, i] < lower_bound) | (X_train[:, i] > upper_bound), i] = np.nan
                X_test[(X_test[:, i] < lower_bound) | (X_test[:, i] > upper_bound), i] = np.nan

            elif outlier_method == 'zscore':
                # Use Mean/Std from TRAIN set (using nan-safe functions)
                mean = np.nanmean(train_col)
                std = np.nanstd(train_col)

                # Avoid division by zero if std is 0
                if std > 0:
                    # Apply outlier masking
                    X_train[np.abs(X_train[:, i] - mean) > 3 * std, i] = np.nan
                    X_test[np.abs(X_test[:, i] - mean) > 3 * std, i] = np.nan

        # Fill missing values using the specified global aggregator derived ONLY from X_train
        for i in range(n_features):
            # Calculate fill value on X_train (after outlier removal)
            if isinstance(fill_missing, (int, float)):
                fill_val = fill_missing
            elif fill_missing == 'mean':
                fill_val = np.nanmean(X_train[:, i])
            elif fill_missing == 'median':
                fill_val = np.nanmedian(X_train[:, i])
            else:
                raise ValueError(f"Unsupported fill_missing method: {fill_missing}")

            # Check if fill_val is NaN (can happen if a column is full of NaNs)
            if np.isnan(fill_val):
                fill_val = 0.0  # Fallback default

            # Apply imputation to NaNs in X_train
            mask_train = np.isnan(X_train[:, i])
            X_train[mask_train, i] = fill_val

            # Apply imputation to NaNs in X_test
            mask_test = np.isnan(X_test[:, i])
            X_test[mask_test, i] = fill_val

        return X_train, X_test

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None, standardize=True, stratify=False):
        """
        Shuffle and partition the static feature matrix into training and testing sets.
        Converts tabular data into 2D NumPy arrays for model ingestion.

        Parameters:
        - test_size (float/int): Fraction or absolute number of samples for testing.
        - shuffle (bool): Whether to randomize sample order.
        - random_state (int): Seed for reproducibility.
        - standardize (bool): Whether to scale features to zero mean and unit variance.
        - stratify (bool): Whether to use stratified sampling based on labels.

        Returns:
        - X_train, X_test, y_train, y_test: Partitioned feature matrices and labels.
        """
        X = self.wide_df[self.features].values
        y = self.y.values

        stratify_param = y if (stratify and shuffle) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify_param
        )

        if standardize:
            self.scaler_2d = StandardScaler()
            X_train = self.scaler_2d.fit_transform(X_train)
            X_test = self.scaler_2d.transform(X_test)

        return X_train, X_test, y_train, y_test
