import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .TimeLSTM import TimeLSTM
from .utils import Classifier


class LSTMNet(nn.Module):
    """
    Standard LSTM network for many-to-one sequence classification.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=2):
        """
        Initialize LSTM network.

        Parameters:
        - input_size: Number of features per time step.
        - hidden_size: Number of hidden units in LSTM cell.
        - num_layers: Number of recurrent layers.
        - num_classes: Number of target output classes.
        """
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
        - Output logits of shape (batch, num_classes).
        """
        # Extract features from the last time step of the LSTM output
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TLSTMNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, cuda_flag, time_index):
        super(TLSTMNet, self).__init__()
        self.time_index = time_index
        # The number of features received by TimeLSTM = total number of features - 1 (excluding the time column)
        self.tlstm = TimeLSTM(input_size - 1, hidden_size, cuda_flag=cuda_flag)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq, feat]
        # Extract the time column while preserving the dimensions [batch, seq].
        ts = x[:, :, self.time_index]
        # Extract other feature columns
        feat_idx = [i for i in range(x.shape[2]) if i != self.time_index]
        feats = x[:, :, feat_idx]
        # T-LSTM calculate
        out = self.tlstm(feats, ts)
        return self.fc(out[:, -1, :])


class TSClassifier(Classifier):

    def __init__(self, model_type='LSTM', hidden_size=64, time_index=None, device=None):
        self.model_type = model_type.upper()
        self.hidden_size = hidden_size
        self.time_index = time_index
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X_train, y_train, epochs=50, lr=0.005):
        X = torch.FloatTensor(X_train).to(self.device)
        y = torch.LongTensor(y_train).to(self.device)
        input_size = X.shape[2]
        num_classes = len(torch.unique(y))

        if self.model_type == "LSTM":
            self.model = LSTMNet(input_size, self.hidden_size, num_classes=num_classes).to(self.device)
        elif self.model_type == "T-LSTM":
            is_cuda = (self.device.type == 'cuda')
            # Passing in the time_index facilitates internal splitting within TLSTMNet.
            self.model = TLSTMNet(input_size, self.hidden_size, num_classes, is_cuda, self.time_index).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        pbar = tqdm(range(epochs), desc=f"Training {self.model_type}")
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = self.model(X)  # Splitting of internal processing X
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    def predict(self, X_test):
        self.model.eval()
        X = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy(), torch.softmax(outputs, dim=1).cpu().numpy()
