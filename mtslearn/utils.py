import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


class Classifier:

    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate model performance and generate visualization reports.

        Parameters:
        - X_test: Test features array.
        - y_test: Ground truth labels.
        - timestamps: Temporal data for T-LSTM.
        - class_names: List of strings for category labels.
        """
        y_pred, y_probs = self.predict(X_test)
        num_classes = y_probs.shape[1]

        print(f"\n--- {self.model_type} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=class_names))
        self._plot_confusion_matrix(y_test, y_pred, class_names)
        self._plot_roc_curve(y_test, y_probs, num_classes)

    def _plot_loss(self, loss_history):
        """Internal helper for loss visualization."""
        plt.figure()
        plt.plot(loss_history, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Internal helper for confusion matrix plotting."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix")
        plt.show()

    def _plot_roc_curve(self, y_true, y_probs, num_classes):
        """Internal helper for ROC-AUC visualization."""
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        plt.figure()
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        else:
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
