""" Contrastive
modiofied and combined together by Jerry
!pip install -q timm pytorch-metric-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        # losses.SupConLoss(temperature=0.1, **kwargs)
        # return losses.NTXentLoss(temperature=self.temperature)(logits, torch.squeeze(labels))

        return losses.SupConLoss(temperature=self.temperature)(logits, torch.squeeze(labels))
        # return losses.ContrastiveLoss(temperature=self.temperature)(logits, torch.squeeze(labels))


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """

    def __init__(self, num_splits=2, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss

# class BinaryCrossEntropy(nn.Module):
#     """ BCE with optional one-hot from dense targets, label smoothing, thresholding
#     NOTE for experiments comparing CE to BCE /w label smoothing, may remove
#     """
#     def __init__(
#             self, smoothing=0.1, target_threshold: float = None, weight: torch.Tensor = None,
#             reduction: str = 'mean', pos_weight: torch.Tensor = None):
#         super(BinaryCrossEntropy, self).__init__()
#         assert 0. <= smoothing < 1.0
#         self.smoothing = smoothing
#         self.target_threshold = target_threshold
#         self.reduction = reduction
#         self.register_buffer('weight', weight)
#         self.register_buffer('pos_weight', pos_weight)
#
#     def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         assert x.shape[0] == target.shape[0]
#         if target.shape != x.shape:
#             # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
#             num_classes = x.shape[-1]
#             # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
#             off_value = self.smoothing / num_classes
#             on_value = 1. - self.smoothing + off_value
#             target = target.long().view(-1, 1)
#             target = torch.full(
#                 (target.size()[0], num_classes),
#                 off_value,
#                 device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
#         if self.target_threshold is not None:
#             # Make target 0, or 1 if threshold set
#             target = target.gt(self.target_threshold).to(dtype=target.dtype)
#         return F.binary_cross_entropy_with_logits(
#             x, target,
#             self.weight,
#             pos_weight=self.pos_weight,
#             reduction=self.reduction)
#

# class SoftTargetCrossEntropy(nn.Module):
#
#     def __init__(self):
#         super(SoftTargetCrossEntropy, self).__init__()
#
#     def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
#         return loss.mean()





