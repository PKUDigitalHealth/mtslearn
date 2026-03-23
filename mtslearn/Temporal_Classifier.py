import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from .utils import TransformerClassifier, CoxModel


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
            y_pred = np.argmax(y_probs_or_preds, axis=-1)
        else:
            y_pred = y_probs_or_preds

        # Flatten sequences to compute overall standard classification metrics
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        print(f"\n--- {self.model_type} Temporal Classification Report (flattened) ---")
        print(classification_report(y_test_flat, y_pred_flat, target_names=class_names))

        # Plot 1: Sample-level misclassification rate distribution
        self._plot_sample_error_distribution(y_test, y_pred)

        # Plot 2: Time-step-level directional error (requires classes to have ordinal meaning)
        self._plot_timestep_error_distribution(y_test, y_pred)

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
        self.DEFAULT_CONFIGS = {
            'CoxPH': {
                'penalizer': 0.1,
                'l1_ratio': 0.0
            }
        }