import pandas as pd
import numpy as np


class Processor:

    def __init__(self):
        self.wide_df = None
        self.id_col = None
        self.time_col = None
        self.label_col = None
        self.features = []

    def read_file(self, num_type, file_path, time_col, id_col, label_col=None, value_col=None, attr_col=None, data_type='long'):
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
        self.num_type = num_type

        if file_path.endswith('.csv') or file_path.endswith('.txt'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df[time_col] = pd.to_datetime(df[time_col])

        if data_type == 'long':
            # Transform multiple rows of metrics into a single row per ID and timestamp
            pivot_idx = [id_col, time_col]
            if label_col:
                pivot_idx.append(label_col)
            self.wide_df = df.pivot_table(index=pivot_idx, columns=attr_col, values=value_col).reset_index()
        else:
            self.wide_df = df

        # Identify feature columns by excluding metadata columns
        exclude = [id_col, time_col, label_col]
        self.features = [c for c in self.wide_df.columns if c not in exclude]

        # Convert time information into floating-point numbers
        date_cols = self.wide_df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if col != self.time_col:
                self.wide_df[col] = self.wide_df[col].astype(np.int64) // 10**9

        if label_col:
            # Map a single ground truth label to each unique ID
            self.y = self.wide_df.groupby(id_col)[label_col].first().values

    def export(self, export_path):
        """
        Save the processed wide-format DataFrame to Excel.

        Parameters:
        - export_path: Path where the output Excel file will be saved.
        """
        self.wide_df.to_excel(export_path, index=False)


class TSProcessor(Processor):

    def data_cleaning(self, fill_missing='mean', outlier_method='iqr'):
        """
        Clean data by handling outliers and imputing missing values.

        Parameters:
        - fill_missing: Method to fill NaNs ('mean', 'median', or 'ffill').
        - outlier_method: Strategy for outlier detection ('iqr', 'zscore', or 'clamp').
        """
        for col in self.features:
            if outlier_method == 'iqr':
                q1, q3 = self.wide_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                # Replace outliers with NaN to prepare for statistical imputation
                self.wide_df.loc[(self.wide_df[col] < q1 - 1.5*iqr) | (self.wide_df[col] > q3 + 1.5*iqr), col] = np.nan

            elif outlier_method == 'zscore':
                mean, std = self.wide_df[col].mean(), self.wide_df[col].std()
                self.wide_df.loc[(self.wide_df[col] - mean).abs() > 3 * std, col] = np.nan

            elif outlier_method == 'clamp':
                # Force values into the 5th and 95th percentile range
                low, high = self.wide_df[col].quantile([0.05, 0.95])
                self.wide_df[col] = self.wide_df[col].clip(low, high)

        self.features.append(self.time_col)
        if fill_missing == 'ffill':
            # Local forward fill followed by global mean for any remaining NaNs
            self.wide_df[self.features] = self.wide_df.groupby(self.id_col)[self.features].ffill()
            self.wide_df[self.features] = self.wide_df[self.features].fillna(self.wide_df[self.features].mean())
            fill_missing = 'mean'

        if fill_missing in ['mean', 'median']:
            # Cascade imputation: fill with group-specific stats first, then global stats
            group_stats = self.wide_df.groupby(self.id_col)[self.features].transform(fill_missing)
            global_stats = getattr(self.wide_df[self.features], fill_missing)()

            self.wide_df[self.features] = self.wide_df[self.features].fillna(group_stats)
            self.wide_df[self.features] = self.wide_df[self.features].fillna(global_stats)

        self.features.remove(self.time_col)
        return self.wide_df

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

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None):
        # 1. 排序确保时序正确
        df_sorted = self.wide_df.sort_values([self.id_col, self.time_col])

        # 2. 计算步间时间间隔 (Time Delta)，第一个点填 0
        temp_df = df_sorted.copy()
        temp_df['time_delta'] = temp_df.groupby(self.id_col)[self.time_col].diff().dt.total_seconds().fillna(0)

        # 记录时间列所在的索引位置（最后一列）
        current_features = self.features + ['time_delta']
        self.time_index = len(current_features) - 1

        # 3. 分组并转换为 3D Tensor
        groups = [group[current_features].values for _, group in temp_df.groupby(self.id_col, sort=False)]
        n_samples = len(groups)
        max_steps = max(len(g) for g in groups)
        n_features = len(current_features)

        X_3d = np.zeros((n_samples, max_steps, n_features))
        for i, g in enumerate(groups):
            X_3d[i, :len(g), :] = g

        # 4. 划分数据集
        n_test = int(n_samples * test_size)
        indices = np.arange(n_samples)
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

        test_idx, train_idx = indices[:n_test], indices[n_test:]
        X_train, X_test = X_3d[train_idx], X_3d[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        return X_train, X_test, y_train, y_test


class StaticProcessor(Processor):

    def extract_features(self, agg_funcs=['mean', 'std', 'max', 'min']):
        """
        Aggregate time-series data into static statistical features for each unique ID.
        Flattens temporal variance into a single row per instance.

        Parameters:
        - agg_funcs (list): List of aggregation strings (e.g., 'mean', 'std', 'max', 'min', 'skew').

        Returns:
        - wide_df (pd.DataFrame): Aggregated DataFrame with flattened feature columns.
        """
        # Group temporal observations by ID and apply multi-statistic aggregation
        agg_df = self.wide_df.groupby(self.id_col)[self.features].agg(agg_funcs)

        # Flatten MultiIndex columns into 'Feature_Function' naming convention
        agg_df.columns = [f"{col}_{func}" for col, func in agg_df.columns]
        self.features = list(agg_df.columns)

        # Synchronize labels by taking the first observation per ID group
        if self.label_col:
            labels = self.wide_df.groupby(self.id_col)[self.label_col].first()
            self.wide_df = pd.concat([agg_df, labels], axis=1).reset_index()
            self.y = self.wide_df[self.label_col].values
        else:
            self.wide_df = agg_df.reset_index()

        return self.wide_df

    def data_cleaning(self, fill_missing='mean', outlier_method='iqr'):
        """
        Perform statistical cleaning on aggregated features.
        Detects outliers via IQR/Z-Score and imputes missing values using global statistics.

        Parameters:
        - fill_missing (str): Imputation method ('mean', 'median').
        - outlier_method (str): Logic for detection ('iqr' or 'zscore').

        Returns:
        - wide_df (pd.DataFrame): The cleaned feature set.
        """
        df = self.wide_df
        cols = self.features

        # Iterate through features to flag and nullify extreme statistical outliers
        for col in cols:
            if outlier_method == 'iqr':
                # Use Interquartile Range to handle skewed distributions
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df.loc[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr), col] = np.nan
            elif outlier_method == 'zscore':
                # Use Standard Deviation threshold for normal distributions
                mean, std = df[col].mean(), df[col].std()
                df.loc[(df[col] - mean).abs() > 3 * std, col] = np.nan

        # Fill missing values using the specified global aggregator on the cleaned columns
        fill_values = getattr(df[cols], fill_missing)()
        df[cols] = df[cols].fillna(fill_values)

        self.wide_df = df
        return self.wide_df

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None):
        """
        Shuffle and partition the static feature matrix into training and testing sets.
        Converts tabular data into 2D NumPy arrays for model ingestion.

        Parameters:
        - test_size (float): Fraction of data for the test subset.
        - shuffle (bool): Whether to randomize sample order.
        - random_state (int): Seed for reproducibility.

        Returns:
        - X_train, X_test, y_train, y_test: Partitioned feature matrices and labels.
        """
        X = self.wide_df[self.features].values
        y = self.y

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Perform index-based shuffling to maintain X-y alignment
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

        # Slice indices based on the calculated split ratio
        n_test = int(n_samples * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
