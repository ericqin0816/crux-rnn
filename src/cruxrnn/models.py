"""Same complete-cue task with Elman, GRU or LSTM recurrence."""

import torch
from torch import nn


class SequenceModel(nn.Module):
    """Map (batch,1,7) cues to (batch,7,4) logits with raw-logit feedback.

    The eighth, unused recurrent transition deliberately preserves v1 timing.
    Different architectures are paired by examples and within-architecture
    initialization, not by claiming that differently shaped weights are equal.
    """

    def __init__(self, architecture: str = "rnn", hidden_size: int = 7):
        super().__init__()
        if (
            architecture not in ("rnn", "gru", "lstm")
            or type(hidden_size) is not int
            or hidden_size < 1
        ):
            raise ValueError("Expected rnn/gru/lstm and a positive integer hidden size")
        self.architecture = architecture
        self.hidden_size = hidden_size
        layer = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[architecture]
        self.recurrence = layer(7, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 4)
        self.feedback = nn.Linear(4, 7)

    def forward(self, cue: torch.Tensor) -> torch.Tensor:
        if cue.ndim != 3 or tuple(cue.shape[1:]) != (1, 7):
            raise ValueError("Expected complete cues shaped (batch,1,7)")
        hidden = cue.new_zeros((1, cue.shape[0], self.hidden_size))
        state = (hidden, hidden.clone()) if self.architecture == "lstm" else hidden
        features, state = self.recurrence(cue, state)
        outputs = []
        for _ in range(7):
            logits = self.readout(features)
            outputs.append(logits)
            features, state = self.recurrence(self.feedback(logits), state)
        return torch.cat(outputs, dim=1)


def parameter_count(model: nn.Module) -> int:
    """Count trainable scalars, including readout and feedback projections."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
