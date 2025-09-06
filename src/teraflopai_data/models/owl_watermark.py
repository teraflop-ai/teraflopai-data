import torch
from torch import nn
from transformers import Owlv2VisionModel


class DetectorModelOwl(nn.Module):
    owl: Owlv2VisionModel

    def __init__(self, model_path: str, dropout: float = 0.0, n_hidden: int = 768):
        super().__init__()

        owl = Owlv2VisionModel.from_pretrained(model_path)
        assert isinstance(owl, Owlv2VisionModel)
        self.owl = owl
        self.owl.requires_grad_(False)
        self.transforms = None

        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
        self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
        self.linear2 = nn.Linear(n_hidden * 2, 2)

    def forward(self, pixel_values: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
            x = outputs.last_hidden_state
            x = self.dropout1(x)
            x = self.ln1(x)
            x = self.linear1(x)
            x = self.act1(x)
            x = self.dropout2(x)
            x, _ = x.max(dim=1)
            x = self.ln2(x)
            x = self.linear2(x)
        return (x,)
