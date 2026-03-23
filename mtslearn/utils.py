import torch
import torch.nn as nn
import torch.optim as optim
from .TimeLSTM import TimeLSTM
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd


class LSTMNet(nn.Module):
    """
    Standard LSTM Architecture for many-to-one sequence classification tasks.
    Maps a sequence of features to a fixed-size category distribution.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=2):
        """
        Parameters:
        - input_size (int): Dimension of the input feature vector per time step.
        - hidden_size (int): The number of features in the LSTM hidden state.
        - num_layers (int): Number of recurrent layers stacked on top of each other.
        - num_classes (int): Number of target categories for the final linear projection.
        """
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Logic: Processes sequence and extracts the last hidden state for classification.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
        - torch.Tensor: Unnormalized logit scores of shape (batch_size, num_classes).
        """
        # LSTM output: (batch, seq_len, hidden_size). Extract only the final step (many-to-one).
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TLSTMNet(nn.Module):
    """
    Time-Aware LSTM (T-LSTM) Architecture.
    Specifically designed to handle irregular time intervals between observations.
    """

    def __init__(self, input_size, hidden_size, num_classes, cuda_flag, time_index):
        """
        Parameters:
        - input_size (int): Total features including the temporal delta column.
        - hidden_size (int): Internal hidden state dimension.
        - num_classes (int): Output classification dimensions.
        - cuda_flag (bool): Whether to enable GPU acceleration for custom T-LSTM cells.
        - time_index (int): Column index of the time-delta feature in the input tensor.
        """
        super(TLSTMNet, self).__init__()
        self.time_index = time_index
        # Reduce input size by 1 as the temporal feature is processed separately by the cell logic.
        self.tlstm = TimeLSTM(input_size - 1, hidden_size, cuda_flag=cuda_flag)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Logic: Decouples temporal data from structural features before T-LSTM computation.

        Parameters:
        - x (torch.Tensor): Shape (batch, seq, feat).

        Returns:
        - torch.Tensor: Classification logits.
        """
        # Slicing: isolate the time-delta channel [batch, seq]
        ts = x[:, :, self.time_index]

        # Slicing: mask out the time-delta channel to get pure features
        feat_idx = [i for i in range(x.shape[2]) if i != self.time_index]
        feats = x[:, :, feat_idx]

        out = self.tlstm(feats, ts)
        return self.fc(out[:, -1, :])


class TorchModelWrapper:
    """
    Universal High-Level Wrapper for PyTorch Modules.
    Standardizes training loops and inference for downstream Classifier orchestration.
    """

    def __init__(self, model_class, **kwargs):
        """
        Parameters:
        - model_class (nn.Module): The class to instantiate.
        - **kwargs: Hyperparameters (lr, epochs, batch_size) and model-specific configs.
        """
        self.model_class = model_class
        self.config = kwargs
        self.model = None
        self.loss_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        """
        Standardized training loop including DataLoader preparation and optimization.

        Parameters:
        - X (torch.Tensor): 3D Input features (batch, seq, feat).
        - y (torch.Tensor): 1D Ground truth labels.

        Returns:
        - self: Trained instance.
        """
        device = self.device

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()  # Explicitly convert to long to ensure CrossEntropy compatibility

        # Infer architectural parameters dynamically from input data shape
        input_size = X.shape[2]
        num_classes = len(torch.unique(y))

        # Extract training-specific parameters while leaving model-specific ones in config
        epochs = self.config.pop('epochs', 10)
        lr = self.config.pop('lr', 0.001)
        batch_size = self.config.pop('batch_size', 32)

        self.model = self.model_class(input_size=input_size, num_classes=num_classes, **self.config).to(device)

        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Log mean loss for convergence monitoring
            self.loss_history.append(total_loss / len(loader))
        return self

    def predict(self, X):
        """
        Parameters:
        - X (torch.Tensor): Input features.

        Returns:
        - tuple: (class_indices, probability_distribution) as numpy arrays.
        """
        X = torch.from_numpy(X).float().to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        return probs.cpu().numpy()


class LSTM(TorchModelWrapper):
    """Factory class for standard LSTM models."""

    def __init__(self, **kwargs):
        super().__init__(LSTMNet, **kwargs)


class TLSTM(TorchModelWrapper):
    """Factory class for Time-Aware LSTM models."""

    def __init__(self, **kwargs):
        super().__init__(TLSTMNet, **kwargs)


class XGB(XGBClassifier):
    """
    Extended XGBoost Classifier.
    Adapts Scikit-Learn style interface to meet the Classifier orchestrated API.
    """

    def fit(self, X, y, **kwargs):
        """
        Extends XGB fit to capture loss history during training.

        Parameters:
        - X (np.ndarray): 2D Features.
        - y (np.ndarray): Labels.
        """
        # Force evaluation on training set to extract metric history without requiring a separate validation set
        super().fit(X, y, eval_set=[(X, y)], verbose=False, **kwargs)

        # Parse evals_result to provide a consistent self.loss_history list for visualization
        results = self.evals_result()
        metric_name = list(results['validation_0'].keys())[0]
        self.loss_history = results['validation_0'][metric_name]
        return self


class TransformerNet(nn.Module):
    """
    Standard Transformer Encoder architecture for sequence labeling tasks.
    """

    def __init__(self, input_size, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerNet, self).__init__()
        # Project input features to the Transformer's latent dimension
        self.embedding = nn.Linear(input_size, d_model)

        # Multi-layer Transformer Encoder for temporal dependency modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head per time step
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Forward pass for the Transformer model.

        Parameters:
        - x (torch.Tensor): Input sequence. Shape: (batch, seq_len, input_size).

        Returns:
        - logits (torch.Tensor): Raw model outputs. Shape: (batch, seq_len, num_classes).
        """
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        logits = self.classifier(x)
        return logits


class TransformerClassifier(TorchModelWrapper):
    """
    Factory wrapper for Transformer-based sequence classification.
    """

    def __init__(self, **kwargs):
        super().__init__(TransformerNet, **kwargs)

    def fit(self, X, y):
        """
        Trains the Transformer model using Cross-Entropy Loss for sequences.

        Parameters:
        - X (np.ndarray): Feature sequences. Shape: (samples, seq_len, features).
        - y (np.ndarray): Target label sequences. Shape: (samples, seq_len).
        """
        device = self.device

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        input_size = X.shape[2]
        num_classes = len(torch.unique(y))

        # Retrieve training hyperparameters from config
        epochs = self.config.pop('epochs', 10)
        lr = self.config.pop('lr', 0.001)
        batch_size = self.config.pop('batch_size', 32)

        self.model = self.model_class(input_size=input_size, num_classes=num_classes, **self.config).to(device)

        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Criterion expects (N, C, L) for sequence tasks
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                out = self.model(batch_x)  # Shape: (Batch, Seq, Classes)

                # Transpose dimensions to (Batch, Classes, Seq) for CrossEntropyLoss compatibility
                loss = criterion(out.transpose(1, 2), batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.loss_history.append(total_loss / len(loader))
        return self


class CoxModel:
    """
    Standardized Wrapper for the Cox Proportional Hazards Model.
    Ensures temporal output alignment with target sequence length.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CoxPHFitter.
        """
        self.fitter = CoxPHFitter(**kwargs)
        self.seq_len_out = None

    def fit(self, X, y):
        """
        Trains the Cox model and records the target sequence length.
        """
        self.seq_len_out = y.shape[1]

        # Event transformation: 1-based indexing for survival durations
        has_event = np.any(y > 0, axis=1)
        durations = np.argmax(y > 0, axis=1) + 1
        durations[~has_event] = self.seq_len_out

        df = pd.DataFrame(X)
        df['duration'] = durations
        df['event'] = has_event.astype(int)

        self.fitter.fit(df, duration_col='duration', event_col='event')

    def predict_proba(self, X):
        """
        Predicts survival probabilities aligned to seq_len_out.
        
        Returns:
        - np.ndarray: Shape (num_samples, seq_len_out, 2).
        """
        # Explicitly define the time points to match seq_len_out (1 to N)
        target_times = np.arange(1, self.seq_len_out + 1)

        # predict_survival_function(X, times=...) ensures output length == len(target_times)
        # Resulting shape after T: (num_samples, seq_len_out)
        surv_func = self.fitter.predict_survival_function(X, times=target_times).values.T

        # P(Class 0) = Survival, P(Class 1) = Event occurred
        p_class_0 = surv_func
        p_class_1 = 1 - p_class_0

        return np.stack([p_class_0, p_class_1], axis=-1)
