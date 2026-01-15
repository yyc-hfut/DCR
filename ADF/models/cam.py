import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, labels=None):
        batch_size, seq_len, _ = features.size()
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)

        weights = self.classifier.weight
        cam = torch.einsum("btd,cd->btc", features, weights)
        cam = cam + self.classifier.bias.view(1, 1, -1)

        time_weights = torch.ones(batch_size, seq_len, device=features.device) / seq_len
        if labels is not None and self.training:
            label_indices = labels.view(-1, 1, 1).expand(-1, seq_len, 1)
            true_cam = cam.gather(dim=2, index=label_indices).squeeze(-1)
            time_weights = F.softmax(true_cam, dim=1)

        weighted_cam = cam * time_weights.unsqueeze(-1)
        soft_labels = F.softmax(weighted_cam, dim=2)

        loss = 0.0
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return soft_labels, logits, loss, time_weights


class RAW_CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes, proj_layer=None, transformer=None, conv_layer=None):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.proj_layer = proj_layer
        self.transformer = transformer
        self.conv_layer = conv_layer

    def forward(self, features, raw_input=None, labels=None):
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        weights = self.classifier.weight
        cam = torch.einsum("btd,cd->btc", features, weights)
        soft_labels = F.softmax(cam, dim=2)

        raw_soft_labels = soft_labels
        raw_cam = None
        if raw_input is not None and labels is not None and all(
            [self.proj_layer, self.transformer, self.conv_layer]
        ):
            x = self.proj_layer(raw_input)
            x = self.transformer(x, attention_mask=None)
            x = x.transpose(1, 2)
            x = self.conv_layer(x)
            x = x.transpose(1, 2)
            pooled = x.mean(dim=1)
            logits = self.classifier(pooled)

            raw_cam = torch.zeros(
                raw_input.shape[0],
                raw_input.shape[1],
                self.classifier.out_features,
                device=raw_input.device,
            )
            for c in range(self.classifier.out_features):
                if raw_input.grad is not None:
                    raw_input.grad.zero_()
                grad_mask = torch.zeros_like(logits)
                grad_mask[:, c] = 1.0
                logits.backward(gradient=grad_mask, retain_graph=True)
                gradients = raw_input.grad
                alpha = gradients.mean(dim=-1, keepdim=True)
                raw_cam[:, :, c : c + 1] = (alpha * raw_input).sum(dim=-1, keepdim=True)
                raw_cam[:, :, c : c + 1] = F.relu(raw_cam[:, :, c : c + 1])

            raw_soft_labels = F.softmax(raw_cam, dim=2)
            raw_input.requires_grad_(False)

        loss = 0.0
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return soft_labels, raw_soft_labels, logits, loss
