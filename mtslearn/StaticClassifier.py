from xgboost import XGBClassifier
from .utils import Classifier


class StaticClassifier(Classifier):
    """
    Classifier wrapper designed for static feature vectors.
    Follows the TSClassifier architecture to instantiate and train models dynamically during fit.
    """

    def __init__(self, model_type='XGB', device='cpu'):
        """
        Initialize the classifier with specific model architecture and hardware target.

        Parameters:
        - model_type (str): Type of model to instantiate (e.g., 'XGB').
        - device (str): Computational device ('cpu' or 'cuda').
        """
        self.model_type = model_type
        self.device = device
        self.model = None

    def fit(self, X_train, y_train, model_config={}):
        """
        Instantiate the underlying model based on type and train on the provided dataset.
        Automatically handles hardware-specific configurations and logs training loss.

        Parameters:
        - X_train (np.ndarray): 2D training feature matrix.
        - y_train (np.ndarray): 1D target labels.
        - model_config (dict): Additional hyperparameters for the model constructor.
        """
        # Select tree construction algorithm based on hardware availability
        tree_method = 'hist' if self.device == 'cuda' else 'auto'

        if self.model_type == 'XGB':
            self.model = XGBClassifier(tree_method=tree_method, device=self.device, **model_config)

            # Execute training with internal evaluation for loss tracking
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

            # Dynamically extract loss history from the first available metric in results
            results = self.model.evals_result()
            data_res = next(iter(results.values()))
            loss_history = next(iter(data_res.values()))
            self._plot_loss(loss_history)
        else:
            # Raise exception for unsupported model types to prevent silent failure
            raise Exception(f"Model type {self.model_type} not implemented")

    def predict(self, X_test):
        """
        Generate class predictions and probability distributions for the test set.

        Parameters:
        - X_test (np.ndarray): 2D feature matrix for inference.

        Returns:
        - predicted, probabilities: NumPy arrays of hard labels and soft scores.
        """
        # Standardized inference interface returning both crisp and probabilistic outputs
        predicted = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        return predicted, probabilities
