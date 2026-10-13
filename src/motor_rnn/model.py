"""One complete sequence cue, followed by seven autoregressive logits."""
import torch
from torch import nn


class MotorRNN(nn.Module):
    def __init__(self, source_exact=True):
        super().__init__()
        self.source_exact = source_exact
        self.recurrence = nn.RNN(7, 7, nonlinearity="tanh", batch_first=True)
        self.readout = nn.Linear(7, 4)
        self.feedback = nn.Linear(4, 7)

    def forward(self, cue):
        if cue.ndim != 3 or tuple(cue.shape[1:]) != (1, 7):
            raise ValueError("Expected a complete sequence cue shaped (batch, 1, 7)")
        hidden = cue.new_zeros((1, cue.shape[0], 7))
        features, hidden = self.recurrence(cue, hidden)
        steps = []
        for position in range(7):
            logits = self.readout(features)
            steps.append(logits)
            if self.source_exact or position < 6:
                # Raw logits, including the source's final unused transition.
                features, hidden = self.recurrence(self.feedback(logits), hidden)
        return torch.cat(steps, dim=1)
