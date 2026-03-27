import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from .utils import TransformerClassifier, CoxModel
import warnings


class Temporal_Classifier:
    """
    Base Orchestrator for Temporal Output Classification Models.
    Provides a unified interface for models that output a sequence of class labels over time.
    """

    def fit(self, X_train, y_train, **model_config):
        """
        Initializes and trains the selected sequence classification model.

        Parameters:
        - X_train (np.ndarray): Training feature set. Shape typically (num_samples, seq_len_in, num_features).
        - y_train (np.ndarray): Target label sequences. Shape typically (num_samples, seq_len_out).
        - model_config (dict): Hyperparameter overrides for the specific model instance.
        """
        model_cls = self.MODELS[self.model_type]

        # Merge default configurations with user-provided overrides
        params = {**self.DEFAULT_CONFIGS.get(self.model_type, {}), **model_config}

        self.model = model_cls(**params)
        self.model.fit(X_train, y_train)

        # Ensure num_classes is recorded for probability processing
        self.num_classes = len(np.unique(y_train))

        # Automated trigger for loss visualization if supported
        if hasattr(self.model, 'loss_history'):
            self._plot_loss(self.model.loss_history)

    def predict(self, X_test):
        """
        Standardizes model inference output for sequence classification.

        Parameters:
        - X_test (np.ndarray): Test feature set.

        Returns:
        - tuple or np.ndarray: (class_predictions, class_probabilities) if supported, 
                               otherwise just class_predictions.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, class_names=None):
        """
        Comprehensive performance assessment for temporal classification models,
        generating overall classification metrics and fine-grained error distribution plots.

        Parameters:
        - X_test (np.ndarray): Test features.
        - y_test (np.ndarray): True target sequences. Shape (num_samples, seq_len_out).
        - class_names (list of str, optional): Human-readable labels for categories.
        """
        y_probs_or_preds = self.predict(X_test)

        # Differentiate between probability outputs and direct class predictions
        if isinstance(y_probs_or_preds, np.ndarray) and y_probs_or_preds.ndim == 3:
            # Shape: (samples, seq_len, num_classes)
            y_probs = y_probs_or_preds
            y_pred = np.argmax(y_probs_or_preds, axis=-1)
        else:
            y_probs = None
            y_pred = y_probs_or_preds

        # Flatten sequences to compute overall standard classification metrics
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        print(f"\n--- {self.model_type} Temporal Classification Report (flattened) ---")
        print(classification_report(y_test_flat, y_pred_flat, target_names=class_names))

        # # Plot 1: Sample-level misclassification rate distribution
        # self._plot_sample_error_distribution(y_test, y_pred)

        # # Plot 2: Time-step-level directional error (requires classes to have ordinal meaning)
        # self._plot_timestep_error_distribution(y_test, y_pred)

        # # Plot 3: Time-step-level AUC, F1, P and R
        self._plot_metrics_over_time(y_test, y_pred, y_probs=y_probs)

    def _plot_loss(self, loss_history):
        """
        Visualizes the objective function convergence over training iterations.

        Parameters:
        - loss_history (list/np.ndarray): Sequential loss values per epoch.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label='Training Loss')
        plt.title('Training Loss Curve (Temporal Classification)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_sample_error_distribution(self, y_true, y_pred):
        """
        Plots a fine-grained histogram of the Misclassification Rate per sample.
        This smooth curve shows how many samples have 0% error, 10% error, etc.

        Parameters:
        - y_true (np.ndarray): Ground truth class sequences. Shape (samples, seq_len).
        - y_pred (np.ndarray): Predicted class sequences. Shape (samples, seq_len).
        """
        # Calculate error rate per sample (0.0 means perfect, 1.0 means completely wrong)
        # axis=1 computes the mean across the sequence length
        sample_error_rates = np.mean(y_pred != y_true, axis=1)

        plt.figure(figsize=(8, 5))
        # Use high number of bins for a continuous 'curve-like' look
        num_bins = 50
        plt.hist(sample_error_rates, bins=num_bins, density=True, alpha=0.5)

        # Overlay a KDE-like outline using the histogram edge
        counts, bins = np.histogram(sample_error_rates, bins=num_bins, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(bin_centers, counts)

        plt.title('Sample-wise Error Distribution (Misclassification Rate)')
        plt.xlabel('Misclassification Rate per Sample (0.0 = Perfect, 1.0 = All Wrong)')
        plt.ylabel('Density')
        plt.show()

    def _plot_timestep_error_distribution(self, y_true, y_pred):
        """
        Visualizes the distribution of directional prediction errors across all time steps.

        Parameters:
        - y_true (np.ndarray): Ground truth class labels.
        - y_pred (np.ndarray): Predicted class labels.
        """
        raw_errors = (y_pred - y_true).flatten()

        plt.figure(figsize=(8, 5))

        # Align bins and bin_centers to ensure consistent discrete intervals for class indices
        limit = self.num_classes - 1
        min_err, max_err = -limit, limit
        # Construct exact boundaries to wrap integer errors (e.g., [-1.5, -0.5, 0.5, 1.5])
        bins = np.arange(min_err - 0.5, max_err + 1.5, 1)

        # Plot density histogram
        plt.hist(raw_errors, bins=bins, density=True)

        # Extract precise density counts and plot the trend line
        counts, _ = np.histogram(raw_errors, bins=bins, density=True)
        bin_centers = np.arange(min_err, max_err + 1)
        plt.plot(bin_centers, counts)

        plt.axvline(x=0, color='black', label='Correct (Error = 0)', linestyle='--')

        plt.title('Time-step-wise Directional Error (Pred_Class - True_Class)')
        plt.xlabel('Class Index Shift (Positive = Over-predicted, Negative = Under-predicted)')
        plt.ylabel('Density')
        plt.xticks(bin_centers)
        plt.legend()
        plt.show()

    def _plot_metrics_over_time(self, y_true, y_pred, y_probs=None, average='macro'):
        """
        Visualizes classification performance stability across sequential time steps.

        This method slices temporal sequence data by time-axis index, computing standard 
        classification metrics (Precision, Recall, F1, and ROC-AUC) for each discrete 
        timestamp. It generates a consolidated line plot to diagnose model performance 
        drift or temporal bias.

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth labels. Shape: (n_samples, n_timesteps).
        y_pred : np.ndarray
            Hard class predictions. Shape: (n_samples, n_timesteps).
        y_probs : np.ndarray, optional (default=None)
            Predicted class probabilities. Shape: (n_samples, n_timesteps, n_classes) 
            for multiclass, or (n_samples, n_timesteps) for binary. Required for AUC.
        average : str, default='macro'
            Aggregation strategy for multiclass metrics. Options: 'micro', 'macro', 
            'samples', 'weighted'.

        Returns:
        --------
        None
            The method renders a Matplotlib figure directly.
        """
        seq_len = y_true.shape[1]

        # Metric accumulators for temporal axis plotting
        precisions, recalls, f1s, aucs = [], [], [], []

        for t in range(seq_len):
            # Isolate cross-sectional slice for current time step 't'
            y_t_true = y_true[:, t]
            y_t_pred = y_pred[:, t]

            # Compute PRF1 metrics; zero_division=0 prevents runtime crashes
            # if a time step contains no instances of a specific class
            p, r, f1, _ = precision_recall_fscore_support(y_t_true, y_t_pred, average=average, zero_division=0)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            if y_probs is not None:
                y_t_probs = y_probs[:, t]
                try:
                    if self.num_classes > 2:
                        # Standard One-vs-Rest (OvR) strategy for multiclass AUC
                        auc = roc_auc_score(y_t_true, y_t_probs, multi_class='ovr', average=average)
                    else:
                        # Extract positive class probabilities (column 1) for binary cases
                        if y_t_probs.ndim > 1 and y_t_probs.shape[1] == 2:
                            prob_for_auc = y_t_probs[:, 1]
                        else:
                            prob_for_auc = y_t_probs
                        auc = roc_auc_score(y_t_true, prob_for_auc)
                    aucs.append(auc)
                except ValueError:
                    # Occurs if y_t_true contains only one unique class at this time step
                    aucs.append(np.nan)
            else:
                aucs.append(np.nan)

        # Plotting Configuration
        plt.figure(figsize=(10, 6))
        time_steps = np.arange(seq_len)

        # Rendering PRF1 curves with distinct markers and line styles
        plt.plot(time_steps, f1s, label=f'F1-score ({average})', color='tab:green', marker='s', linestyle='-', alpha=0.7)
        plt.plot(time_steps, precisions, label=f'Precision ({average})', color='tab:blue', marker='^', linestyle='-.', alpha=0.7)
        plt.plot(time_steps, recalls, label=f'Recall ({average})', color='tab:orange', marker='v', linestyle='--', alpha=0.7)

        if y_probs is not None and not np.all(np.isnan(aucs)):
            # Filter NaN values to allow continuous line plotting for valid AUC segments
            valid_idx = ~np.isnan(aucs)
            plt.plot(
                time_steps[valid_idx],
                np.array(aucs)[valid_idx],
                label=f'ROC-AUC ({average})',
                color='tab:red',
                marker='o',
                linestyle=':',
                alpha=0.7
            )

            if not np.all(valid_idx):
                warnings.warn("Certain time steps with single-class ground truth were omitted from AUC plotting.")

        # Chart aesthetics and axis normalization
        plt.title('Performance Metrics Over Time Steps', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


class Temporal_Temporal_Classifier(Temporal_Classifier):
    """
    Subclass tailored for Temporal-Input to Temporal-Output classification models 
    (e.g., Seq2Seq Classifiers, Temporal Convolutional Networks).
    """

    def __init__(self, model_type='Seq2SeqClassifier'):
        """
        Parameters:
        - model_type (str): Identifier for the model architecture.
        """
        self.model_type = model_type
        self.model = None

        self.MODELS = {
            'Transformer': TransformerClassifier,
        }

        self.DEFAULT_CONFIGS = {'Transformer': {'d_model': 64, 'nhead': 4, 'epochs': 20, 'batch_size': 32, 'lr': 0.001}}


class Static_Temporal_Classifier(Temporal_Classifier):
    """
    Orchestrator for models taking static inputs and producing temporal outputs.
    Fully leverages Temporal_Classifier base logic.
    """

    def __init__(self, model_type='CoxPH'):
        """
        Parameters:
        - model_type (str): Identifier for the static-to-temporal model.
        - seq_len_out (int): Length of the output temporal sequence.
        """
        self.model_type = model_type
        self.model = None

        # Registry for static-temporal models
        self.MODELS = {
            'CoxPH': CoxModel,
        }

        # Default configurations for integrated models
        self.DEFAULT_CONFIGS = {'CoxPH': {'penalizer': 0.1, 'l1_ratio': 0.0}}
