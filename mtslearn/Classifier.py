import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from .utils import LSTM, TLSTM, XGB


class Classifier:
    """
    Base Classifier Orchestrator.
    Provides a unified interface for training, predicting, and evaluating 
    diverse machine learning models including Deep Learning and Gradient Boosting.
    """

    def fit(self, X_train, y_train, model_config={}):
        """
        Initializes and trains the selected model architecture.

        Parameters:
        - X_train (np.ndarray): Training feature set.
        - y_train (np.ndarray): Target labels.
        - model_config (dict): Hyperparameter overrides for the specific model instance.
        """
        model_cls = self.MODELS[self.model_type]

        # Merge default configurations with user-provided overrides via dictionary unpacking
        params = {**self.DEFAULT_CONFIGS.get(self.model_type, {}), **model_config}

        self.model = model_cls(**params)
        self.model.fit(X_train, y_train)

        # Automated trigger for loss visualization if the model records convergence history
        if hasattr(self.model, 'loss_history'):
            self._plot_loss(self.model.loss_history)

    def predict(self, X_test):
        """
        Standardizes model inference output across different framework backends.

        Parameters:
        - X_test (np.ndarray): Test feature set.

        Returns:
        - tuple: (class_predictions, class_probabilities)
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, class_names=None):
        """
        Comprehensive performance assessment generating metrics and visual diagnostic plots.

        Parameters:
        - X_test (np.ndarray): Test features.
        - y_test (np.ndarray): True labels.
        - class_names (list of str, optional): Human-readable labels for categories.
        """
        y_pred, y_probs = self.predict(X_test)
        num_classes = y_probs.shape[1]

        print(f"\n--- {self.model_type} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=class_names))

        self._plot_confusion_matrix(y_test, y_pred, class_names)
        self._plot_roc_curve(y_test, y_probs, num_classes)

    def _plot_loss(self, loss_history):
        """
        Visualizes the objective function convergence over training iterations.

        Parameters:
        - loss_history (list/np.ndarray): Sequential loss values per epoch.
        """
        plt.figure()
        plt.plot(loss_history, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plots a heatmap of prediction accuracy vs. ground truth.

        Parameters:
        - y_true (np.ndarray): Ground truth labels.
        - y_pred (np.ndarray): Model predictions.
        - class_names (list): Labels for axes.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix")
        plt.show()

    def _plot_roc_curve(self, y_true, y_probs, num_classes):
        """
        Computes and visualizes Receiver Operating Characteristic curves.
        Supports binary and multi-class (One-vs-Rest) scenarios.

        Parameters:
        - y_true (np.ndarray): Ground truth labels.
        - y_probs (np.ndarray): Predicted class probabilities.
        - num_classes (int): Total count of unique classes.
        """
        # Binarize labels to calculate ROC for multi-class OvR strategy
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        plt.figure()
        if num_classes == 2:
            # Binary classification uses the probability of the positive class (index 1)
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        else:
            # Multi-class: iterate through each class for individual ROC curves
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line for random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()


class StaticClassifier(Classifier):
    """
    Subclass for models that do not require temporal sequence handling (e.g., XGBoost).
    """

    def __init__(self, model_type='XGB'):
        """
        Parameters:
        - model_type (str): Identifier for the model architecture.
        """
        self.model_type = model_type
        self.model = None
        self.MODELS = {'XGB': XGB}
        self.DEFAULT_CONFIGS = {'XGB': {}}


class TSClassifier(Classifier):
    """
    Subclass for Time-Series specific models requiring sequential data handling.
    """

    def __init__(self, model_type='LSTM', time_index=0):
        """
        Parameters:
        - model_type (str): 'LSTM' or 'T-LSTM'.
        - time_index (int): Column index representing temporal deltas for T-LSTM logic.
        """
        self.model_type = model_type
        self.time_index = time_index
        self.model = None
        self.MODELS = {'LSTM': LSTM, 'T-LSTM': TLSTM}

        self.DEFAULT_CONFIGS = {
            'LSTM': {
                'hidden_size': 128,
                'num_layers': 1
            },
            'T-LSTM': {
                'hidden_size': 128,
                'time_index': self.time_index,  # Inject context-specific time index
                'cuda_flag': False
            }
        }
