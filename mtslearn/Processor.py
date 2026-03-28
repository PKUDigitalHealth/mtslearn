import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings


class Processor:
    """
    A high-level data processing engine designed to transform diverse raw data formats 
    (long, flattened, wide) into a unified structured format suitable for machine learning 
    or time-series analysis.
    """

    def __init__(self):
        """
        Initializes the Processor with placeholders for data storage and metadata.
        """
        self.wide_df = None
        self.id_col = None
        self.time_col = None
        self.label_col = None
        self.label_type = None
        self.y = None
        self.features = []

    def _load_raw_df(self, file_path):
        """
        Internal utility to automatically identify file extensions and load the raw DataFrame.

        Parameters:
        - file_path (str): The system path to the data file.

        Returns:
        - pd.DataFrame: Loaded raw data.
        """
        if file_path.endswith('.psv'):
            return pd.read_csv(file_path, sep='|')
        elif file_path.endswith('.csv') or file_path.endswith('.txt'):
            return pd.read_csv(file_path)
        else:
            # Fallback for Excel formats (.xlsx, .xls)
            return pd.read_excel(file_path)

    def _process_dataframe(self, df, data_type, time_col, id_col, label_col, value_col, attr_col):
        """
        Core transformation engine that pivots or reshapes data into a standardized wide format.

        Parameters:
        - df (pd.DataFrame): The raw input DataFrame.
        - data_type (str): Format of input data ('long', 'flattened', or 'wide').
        - time_col (str): Column name representing the temporal dimension.
        - id_col (str): Column name for the primary entity identifier.
        - label_col (str, optional): Column name for the target/label.
        - value_col (str, optional): The column containing measurements (for 'long' format).
        - attr_col (str, optional): The column containing attribute names (for 'long' format).

        Returns:
        - pd.DataFrame: Reshaped wide-format DataFrame.
        """
        processed_df = None

        if data_type == 'long':
            # Pivot long-format data where multiple rows represent different attributes of the same timestamp
            pivot_idx = [id_col, time_col]
            if label_col:
                pivot_idx.append(label_col)
            processed_df = df.pivot_table(index=pivot_idx, columns=attr_col, values=value_col).reset_index()

        elif data_type == 'flattened':
            # Handle non-standard 'flattened' structures by unrolling interleaved attribute/time pairs
            obs_cols = df.columns[1:-1]
            rows = []
            for _, row in df.iterrows():
                for i in range(0, len(obs_cols), 2):
                    attr_name, attr_val, time_val = obs_cols[i], row[obs_cols[i]], row[obs_cols[i + 1]]
                    # Filter out incomplete observations
                    if pd.notna(attr_val) and pd.notna(time_val):
                        rows.append({
                            id_col: row[id_col],
                            label_col: row[label_col],
                            time_col: time_val,
                            "__attr_name__": attr_name,
                            "__value__": attr_val
                        })
            temp_df = pd.DataFrame(rows)
            # Re-pivot the unrolled rows into a unified wide format
            processed_df = temp_df.pivot_table(index=[id_col, time_col, label_col], columns="__attr_name__",
                                               values="__value__").reset_index()
            processed_df.columns.name = None
        else:
            # Assume data is already in wide format
            processed_df = df

        return processed_df

    def read_file(
        self,
        file_path,
        time_col,
        id_col,
        label_col=None,
        value_col=None,
        attr_col=None,
        data_type='long',
        label_type='static',
        cols_to_remove=None,
        custom_mapping=None,
        datetime_cols=None,
        encode_strings=True
    ):
        """
        Entry point for processing a single data file.

        Parameters:
        - file_path (str): Path to the source file.
        - time_col (str): Name of the timestamp column.
        - id_col (str): Name of the identifier column.
        - label_col (str, optional): Target column name.
        - value_col/attr_col (str, optional): Required if data_type is 'long'.
        - data_type (str): 'long', 'flattened', or 'wide'. Default 'long'.
        - label_type (str): 'static' (one label per ID) or 'dynamic' (label per timestamp).
        - cols_to_remove (list, optional): List of column names to drop.
        - custom_mapping (dict, optional): Dictionary for manual value replacement {col: {old: new}}.
        - datetime_cols (list, optional): List of additional columns to parse as dates.
        - encode_strings (bool): Whether to automatically label-encode categorical features.
        """
        self.id_col, self.time_col, self.label_col, self.label_type = id_col, time_col, label_col, label_type
        raw_df = self._load_raw_df(file_path)

        self.wide_df = self._process_dataframe(raw_df, data_type, time_col, id_col, label_col, value_col, attr_col)
        self._finalize_processing(cols_to_remove, custom_mapping, datetime_cols, encode_strings)

    def read_directory(
        self,
        dir_path,
        time_col,
        id_col=None,
        label_col=None,
        value_col=None,
        attr_col=None,
        data_type='wide',
        label_type='static',
        cols_to_remove=None,
        custom_mapping=None,
        datetime_cols=None,
        encode_strings=True
    ):
        """
        Batch processing mode for a directory containing multiple entity files.

        Parameters:
        - dir_path (str): Path to directory.
        - id_col (str, optional): If None, filenames (minus extension) are used as IDs.
        - (Other params): Identical to read_file.
        """
        self.time_col, self.label_col, self.label_type = time_col, label_col, label_type
        self.id_col = id_col if id_col else "ID"

        all_dfs = []
        extensions = ('.csv', '.psv', '.xlsx', '.xls')
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(extensions) and not f.startswith('.')]

        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            raw_df = self._load_raw_df(file_path)

            # Auto-generate ID from filename if missing in the dataset
            if self.id_col not in raw_df.columns:
                raw_df[self.id_col] = os.path.splitext(file_name)[0]

            individual_wide = self._process_dataframe(
                raw_df, data_type, self.time_col, self.id_col, self.label_col, value_col, attr_col
            )
            all_dfs.append(individual_wide)

        # Merge all entities into a single global wide DataFrame
        self.wide_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        self._finalize_processing(cols_to_remove, custom_mapping, datetime_cols, encode_strings)

    def _finalize_processing(self, cols_to_remove=None, custom_mapping=None, datetime_cols=None, encode_strings=True):
        """
        Unified post-processing pipeline for cleaning, mapping, and encoding the Wide DataFrame.
        """

        # 1. Feature pruning
        if cols_to_remove:
            # Use intersection to avoid KeyErrors during bulk removal
            to_drop = [c for c in cols_to_remove if c in self.wide_df.columns]
            self.wide_df.drop(columns=to_drop, inplace=True)

        # 2. Domain-specific value mapping
        if custom_mapping:
            for col, mapping_dict in custom_mapping.items():
                if col in self.wide_df.columns:
                    self.wide_df[col] = self.wide_df[col].replace(mapping_dict)

        # 3. Temporal normalization
        # Parse main time column; 'coerce' handles malformed dates by returning NaT
        self.wide_df[self.time_col] = pd.to_datetime(self.wide_df[self.time_col], errors='coerce')

        if datetime_cols:
            for col in datetime_cols:
                if col in self.wide_df.columns and col != self.time_col:
                    self.wide_df[col] = pd.to_datetime(self.wide_df[col], errors='coerce')

        # Convert secondary datetime columns to Unix epoch (integers) for model compatibility
        date_cols = self.wide_df.select_dtypes(include=['datetime64', 'datetimetz']).columns
        for col in date_cols:
            if col != self.time_col:
                self.wide_df[col] = self.wide_df[col].astype(np.int64) // 10**9

        # 4. Feature identification
        exclude = [self.id_col, self.time_col, self.label_col]
        self.features = [c for c in self.wide_df.columns if c not in exclude]

        # 5. Categorical encoding
        if encode_strings:
            string_cols = self.wide_df[self.features].select_dtypes(include=['object', 'string', 'category']).columns
            for col in string_cols:
                # factorize performs label encoding; preserves NaNs as -1
                self.wide_df[col] = pd.factorize(self.wide_df[col])[0]

        # 6. Label extraction (y)
        if self.label_col and self.label_col in self.wide_df.columns:
            if self.label_type == 'static':
                # Map one label per ID (takes the first encountered value)
                self.y = self.wide_df.groupby(self.id_col)[self.label_col].first()
            else:
                # Dynamic labels for each time point
                self.y = self.wide_df[self.label_col]

    def export(self, export_path):
        """
        Saves the processed wide DataFrame to an Excel file.

        Parameters:
        - export_path (str): Destination file path.
        """
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        self.wide_df.to_excel(export_path, index=False)

    def load_dataset(self, name):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if name == "sepsis-A":
            data_config = {
                "dir_path": BASE_DIR + '/data/sepsis_data/set_A/',
                "data_type": 'wide',
                "time_col": 'ICULOS',
                "label_col": 'SepsisLabel',
                "label_type": 'temporal'
            }
            self.read_directory(**data_config)
        elif name == "sepsis-B":
            data_config = {
                "dir_path": BASE_DIR + '/data/sepsis_data/set_B/',
                "data_type": 'wide',
                "time_col": 'ICULOS',
                "label_col": 'SepsisLabel',
                "label_type": 'temporal'
            }
            self.read_directory(**data_config)
        elif name == "COVID-19":
            data_config = {
                "file_path": BASE_DIR + '/data/covid19_data/375_patients_example.xlsx',
                "data_type": 'wide',
                "time_col": 'RE_DATE',
                "id_col": 'PATIENT_ID',
                "label_col": 'outcome',
            }
            self.read_file(**data_config)


class TSProcessor(Processor):

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None, stratify=False, max_len=None):
        """
        Splits the processed wide DataFrame into training and testing tensors, handling 
        sequence alignment and optional stratification.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - shuffle (bool): Whether to shuffle the data before splitting.
        - random_state (int, optional): Seed for reproducible shuffling.
        - stratify (bool): If True, performs stratified splitting based on self.y.
        - max_len (int, optional): Maximum sequence length for the output tensors. 
                                   If None, uses the maximum length found in the data.

        Returns:
        - X_train (np.ndarray): Training features of shape (n_train, max_len, n_features).
        - X_test (np.ndarray): Testing features of shape (n_test, max_len, n_features).
        - y_train (np.ndarray): Training labels (1D for static, 2D for temporal).
        - y_test (np.ndarray): Testing labels (1D for static, 2D for temporal).
        """
        # Ensure temporal consistency before processing
        df_sorted = self.wide_df.sort_values([self.id_col, self.time_col])

        # Calculate time intervals between successive observations per entity
        df_sorted['time_delta'] = df_sorted.groupby(self.id_col)[self.time_col].diff().dt.total_seconds().fillna(0)

        current_features = self.features + ['time_delta']
        self.time_index = len(current_features) - 1

        unique_ids = df_sorted[self.id_col].unique()
        is_temporal_label = self.label_type == 'temporal'

        # Extract labels for stratification if applicable (only supported for static labels)
        stratify_labels = self.y.loc[unique_ids].values if (stratify and not is_temporal_label) else None

        # Split at the entity (ID) level to prevent data leakage across time steps
        from sklearn.model_selection import train_test_split as skl_split
        train_ids, test_ids = skl_split(
            unique_ids,
            test_size=test_size,
            random_state=random_state if shuffle else None,
            shuffle=shuffle,
            stratify=stratify_labels if stratify and shuffle else None
        )

        # Determine target sequence length for tensor padding/truncation
        actual_max_steps = df_sorted.groupby(self.id_col).size().max()
        target_len = max_len if max_len is not None else actual_max_steps
        n_features = len(current_features)

        def build_tensors(id_list):
            """
            Internal helper to project grouped DataFrame rows into 3D NumPy tensors.
            """
            X_tensor = np.zeros((len(id_list), target_len, n_features))
            y_tensor = np.zeros((len(id_list), target_len)) if is_temporal_label else np.zeros((len(id_list),))
            times_list = []

            # Grouping once to optimize lookup speed
            grouped = dict(list(df_sorted.groupby(self.id_col)))
            for i, _id in enumerate(id_list):
                group_data = grouped[_id]
                data_values = group_data[current_features].values
                time_values = group_data[self.time_col].values

                seq_len = len(data_values)
                fill_len = min(seq_len, target_len)

                # Implementation Detail: Post-alignment/truncation
                # We take the LAST 'fill_len' steps and place them at the start of the tensor
                X_tensor[i, :fill_len, :] = data_values[-fill_len:]
                times_list.append(time_values[-fill_len:])

                if self.label_col:
                    if is_temporal_label:
                        y_tensor[i, :fill_len] = group_data[self.label_col].values[-fill_len:]
                    else:
                        y_tensor[i] = self.y.loc[_id]

            return X_tensor, y_tensor, times_list

        X_train, y_train, self.train_times = build_tensors(train_ids)
        X_test, y_test, self.test_times = build_tensors(test_ids)

        return X_train, X_test, y_train, y_test

    def time_resample(self, X_train, X_test, freq='1D', fill_method='linear'):
        """
        Resamples temporal features to a fixed frequency (e.g., Daily, Hourly) 
        and handles missing values via interpolation.

        Parameters:
        - X_train (np.ndarray): Training feature tensor from train_test_split.
        - X_test (np.ndarray): Testing feature tensor from train_test_split.
        - freq (str): Pandas offset alias for the target frequency (e.g., '1H', '1D').
        - fill_method (str): Interpolation logic ('linear', 'time', 'nearest').

        Returns:
        - X_train_resampled (np.ndarray): Resampled training features.
        - X_test_resampled (np.ndarray): Resampled testing features (aligned to training length).
        """

        def _process_set(X, times_list, target_len=None):
            """
            Internal helper to apply resampling logic to individual sequence samples.
            """
            resampled_list = []
            n_samples = X.shape[0]
            n_features = X.shape[2]

            for i in range(n_samples):
                # Only process the non-padded portion of the tensor
                valid_len = len(times_list[i])
                sample_data = X[i, :valid_len, :]

                # Map back to datetime index for pandas resampling capabilities
                df = pd.DataFrame(sample_data)
                df.index = pd.to_datetime(times_list[i])

                # Execute frequency conversion and fill gaps
                resampled = df.resample(freq).mean()
                resampled = resampled.interpolate(method=fill_method).ffill().bfill()
                resampled_list.append(resampled.values)

            # Determine global max length after resampling to ensure tensor uniformity
            if target_len is None:
                target_len = max(len(s) for s in resampled_list)

            X_out = np.zeros((n_samples, target_len, n_features))
            for i, s in enumerate(resampled_list):
                fill_len = min(len(s), target_len)
                # Front-alignment padding logic
                X_out[i, :fill_len, :] = s[:fill_len]

            return X_out, target_len

        # Resample train set first to establish the baseline sequence length
        X_train_resampled, train_max_len = _process_set(X_train, self.train_times)
        # Resample test set using the train set length to ensure model input compatibility
        X_test_resampled, _ = _process_set(X_test, self.test_times, target_len=train_max_len)

        return X_train_resampled, X_test_resampled

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

    def scale_features(self, X_train, X_test, method='standardize'):
        """
        Scales 3D tensors features while preserving zero-padding.
        
        Parameters:
        - X_train, X_test (np.ndarray): 3D Tensors [Samples, TimeSteps, Features].
        - method (str): Scaling method, 'standardize' or 'normalize'.

        Returns:
        - X_train_scaled, X_test_scaled (np.ndarray): Scaled 3D Tensors.
        """
        if method not in ['standardize', 'normalize', None]:
            raise ValueError("Method must be 'standardize', 'normalize', or None.")

        if method is None:
            return X_train, X_test

        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'normalize':
            scaler = MinMaxScaler()

        # Copy to avoid inplace modification side-effects
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # --- 3D Padding Handling (Masking) ---
        # Find the real data location that is not padding, mask shape: [Samples, TimeSteps]
        train_mask = ~np.all(X_train == 0, axis=-1)
        test_mask = ~np.all(X_test == 0, axis=-1)

        # Extract only the real data (convert to 2D: [Valid_TimeSteps, Features]) for Fit & Transform
        # This avoids 0 being treated as an actual feature in the statistics and ensures that the padding remains 0 after scaling.
        X_train_scaled[train_mask] = scaler.fit_transform(X_train[train_mask])

        # Use the scaler of the training set to transform the test set to prevent data leakage.
        X_test_scaled[test_mask] = scaler.transform(X_test[test_mask])

        self.scaler = scaler

        return X_train_scaled, X_test_scaled


class StaticProcessor(Processor):

    def extract_features(self, agg_funcs=['mean', 'std', 'max', 'min'], include_duration=False):
        """
        Transforms time-series data into fixed-length statistical summaries. This reduces 
        temporal sequences into global descriptors for standard machine learning models.

        Parameters:
        - agg_funcs (list): List of pandas-compatible aggregation strings to apply to features.
        - include_duration (bool): If True, calculates the total timespan (max - min) for each ID.

        Returns:
        - pd.DataFrame: The restructured wide_df containing aggregated features.
        """
        # 1. Compute global statistical descriptors per entity ID
        agg_df = self.wide_df.groupby(self.id_col)[self.features].agg(agg_funcs)
        # Flatten MultiIndex columns into a single-level naming convention (e.g., 'feature_mean')
        agg_df.columns = [f"{col}_{func}" for col, func in agg_df.columns]

        if include_duration:
            duration = self.wide_df.groupby(self.id_col)[self.time_col].agg(lambda x: x.max() - x.min())
            # Convert timedelta objects to raw seconds for numeric processing
            agg_df['duration'] = duration.dt.total_seconds()

        new_feature_names = list(agg_df.columns)

        # 2. Restructure DataFrame based on the target label architecture
        if self.label_type == 'dynamic':
            # Dynamic Mode: Preserves all time points but attaches global entity statistics to each row.
            # Useful for models that require global context at every local time step.
            self.wide_df = self.wide_df[[self.id_col, self.time_col,
                                         self.label_col]].merge(agg_df.reset_index(), on=self.id_col, how='left')
        else:
            # Static Mode: Compresses the dataset so each ID occupies exactly one row.
            labels = self.wide_df.groupby(self.id_col)[self.label_col].first()
            self.wide_df = pd.concat([agg_df, labels], axis=1).reset_index()

        # Update metadata to reflect the new feature space
        self.features = new_feature_names
        self.y = self.wide_df[self.label_col]
        return self.wide_df

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None, stratify=False):
        """
        Partition features and labels into training and testing sets. Supports standard 
        tabular splitting and sequence-padding for dynamic label sets.

        Parameters:
        - test_size (float): The proportion of data to allocate to the test set.
        - shuffle (bool): Whether to randomize the indices before splitting.
        - random_state (int, optional): Seed for deterministic shuffling.
        - stratify (bool): Whether to ensure balanced label distribution (Static mode only).

        Returns:
        - X_train, X_test (np.ndarray): Feature matrices.
        - y_train, y_test (np.ndarray): Label vectors or matrices.
        """
        # Logic for Static Labels: Standard row-wise splitting
        if self.label_type == 'static':
            X = self.wide_df[self.features].values
            y = self.y.values
            stratify_param = y if (stratify and shuffle) else None
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify_param
            )
        else:
            # Logic for Dynamic Labels: Entity-level splitting to prevent temporal leakage
            unique_ids = self.wide_df[self.id_col].unique()
            # Split the unique IDs first; ensures all timepoints for a single ID stay in the same set
            train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state, shuffle=shuffle)

            # 1. Construct X: Retrieve the pre-computed static features for each ID
            # drop_duplicates ensures we only have one row of statistics per entity
            df_unique = self.wide_df.drop_duplicates(subset=[self.id_col]).set_index(self.id_col)
            X_train = df_unique.loc[train_ids, self.features].values
            X_test = df_unique.loc[test_ids, self.features].values

            # 2. Construct y: Extract label sequences and pad to uniform length
            max_len = self.wide_df.groupby(self.id_col).size().max()

            def build_2d_y(id_list):
                """
                Internal utility to convert group-wise labels into a 2D tensor of shape (n_ids, max_len).
                """
                y_tensor = np.zeros((len(id_list), max_len))
                # Dictionary lookup for O(1) group access during iteration
                grouped = dict(list(self.wide_df.groupby(self.id_col)))
                for i, _id in enumerate(id_list):
                    vals = grouped[_id][self.label_col].values
                    # Sequence Alignment: Post-alignment (takes latest fill_len points)
                    fill_len = min(len(vals), max_len)
                    y_tensor[i, :fill_len] = vals[-fill_len:]
                return y_tensor

            y_train = build_2d_y(train_ids)
            y_test = build_2d_y(test_ids)

            return X_train, X_test, y_train, y_test

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

    def scale_features(self, X_train, X_test, method='standard'):
        """
        Scale features using specified method.

        Parameters:
        - X_train, X_test: Feature matrices.
        - method (str): 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).

        Returns:
        - X_train, X_test: Scaled feature matrices.
        """
        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'normalize':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scale method: '{method}'. Use 'standardize' or 'normalize'.")

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test
