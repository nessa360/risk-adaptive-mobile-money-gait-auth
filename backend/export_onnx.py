import torch
import torch.nn as nn

class GaitCNNLSTM(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.transpose(1, 2)
        lstm_out, _ = self.lstm(feat)
        return self.fc(lstm_out[:, -1, :])

model = GaitCNNLSTM()
model.eval()

dummy_input = torch.randn(1, 6, 128, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    "gait_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["sensor_window"],
    output_names=["match_score_logit"],
    dynamic_axes={
        "sensor_window": {0: "batch_size"},
        "match_score_logit": {0: "batch_size"}
    }
)
print("gait_model.onnx exported successfully!")