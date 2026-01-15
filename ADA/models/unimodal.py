import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, input_dim=None, dropout=0.1):
        super().__init__()
        if input_dim is None:
            self.proj = nn.LazyLinear(hidden_dim)
        else:
            self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _pool(self, x):
        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def forward(self, x):
        pooled = self._pool(x)
        features = self.proj(pooled)
        features = F.relu(self.dropout(features))
        logits = self.classifier(features)
        return logits, features


class TextClassifier(_BaseClassifier):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        use_precomputed=False,
        input_dim=None,
        target_seq_len=None,
        dropout=0.1,
    ):
        super().__init__(hidden_dim, num_classes, input_dim=input_dim, dropout=dropout)
        self.use_precomputed = use_precomputed
        self.target_seq_len = target_seq_len


class AudioClassifier(_BaseClassifier):
    def __init__(self, hidden_dim, num_classes, input_dim=None, dropout=0.1):
        super().__init__(hidden_dim, num_classes, input_dim=input_dim, dropout=dropout)


class VideoClassifier(_BaseClassifier):
    def __init__(self, hidden_dim, num_classes, input_dim=None, dropout=0.1):
        super().__init__(hidden_dim, num_classes, input_dim=input_dim, dropout=dropout)
