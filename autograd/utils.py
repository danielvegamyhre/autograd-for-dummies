'''
Utility functions (loss, regularization, etc.)
'''
from typing import List
from autograd.engine import Scalar
from autograd import nn

def l2_regularization(model: nn.Module, alpha=1e-4) -> float:
    return alpha * sum((p*p for p in model.parameters()))

def svm_max_margin_loss(outputs: List[Scalar], labels: List[int]) -> List[Scalar]:
    return [(1 + -y_i*output_i).relu() for y_i, output_i in zip(labels, outputs)]