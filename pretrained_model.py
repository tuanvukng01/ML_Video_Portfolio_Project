import torch
import torch.nn as nn
import compressai
from compressai.zoo import bmshj2018_factorized

class CompressAIWrapper(nn.Module):
    def __init__(self, quality=3):
        super().__init__()
        # Load bmshj2018_factorized model from CompressAI
        self.model = bmshj2018_factorized(quality=quality, pretrained=True)

    def forward(self, x):
        out = self.model(x)
        return out["x_hat"], out["likelihoods"]