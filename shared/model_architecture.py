import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=32,
        num_layers=1,
        num_classes=4
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # Output 4 classes:
        # 0 = Stable
        # 1 = Medium Risk
        # 2 = High Risk
        # 3 = Critical
        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):
        out, _ = self.lstm(x)

        # Take the last time step output
        out = out[:, -1, :]

        # Return raw logits.
        # Do NOT use sigmoid/softmax here.
        # CrossEntropyLoss will handle this during training.
        out = self.fc(out)

        return out