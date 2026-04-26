import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

import data
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import sys
from ..isc_model.model import BCEMetric

class OjaOneHotModel(nn.Module):
    def __init__(self, lr: float = 10e-3):
        super().__init__()

        self.lr = lr

        self.hidden = nn.Linear(2,1, bias=False)

        nn.init.uniform_(self.hidden.weight,a=-.01,b=.01)

        for param in self.parameters():
            param.requires_grad = False

    
    def forward(self, x: torch.Tensor):
        return self.hidden(x)
    
    def learn_oja(self, x: torch.Tensor, epochs: int = 100) -> pd.DataFrame:
        n = x.shape[0]

        x_a = torch.tensor([[1.0, 0.0]], device=x.device)
        x_b = torch.tensor([[0.0, 1.0]], device=x.device)

        history = {
            "step": [],
            "w1": [],
            "w2": [],
            "y_a": [],
            "y_b": [],
            "sign_gate": [],
            "separation": [],
        }

        step = 0

        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n):
                # Oja update
                datapt = x[perm[i]].unsqueeze(0)
                y = self(datapt)
                dw = y * (datapt - y * self.hidden.weight)

                with torch.no_grad():
                    self.hidden.weight += self.lr * dw

                # track statistics

                y_a = self(x_a).item()
                y_b = self(x_b).item()

                sign_gate = (y_a * y_b) < 0
                separation = abs(y_a - y_b)

                history["step"].append(step)
                history["w1"].append(self.hidden.weight[0,0].item())
                history["w2"].append(self.hidden.weight[0,1].item())
                history["y_a"].append(y_a)
                history["y_b"].append(y_b)
                history["sign_gate"].append(bool(sign_gate))
                history["separation"].append(separation)
                
                step += 1
        
        return pd.DataFrame(history)
    
class OjaEmbedModel(nn.Module):
    def __init__(self, lr: float = 10e-3):
        super().__init__()

        self.lr = lr

        self.input_to_embedding = nn.Linear(5,16,bias=False)
        self.embedding_to_output = nn.Linear(16,1,bias=False)

        nn.init.uniform_(self.input_to_embedding.weight,a=-.01,b=.01)
        nn.init.uniform_(self.embedding_to_output.weight,a=-.01,b=.01)

        for param in self.parameters():
            param.requires_grad = False

    
    def forward(self, x: torch.Tensor):
        embedding = self.input_to_embedding(x)
        output = self.embedding_to_output(embedding)
        return output
    
    def learn_oja(self, x: torch.Tensor, epochs: int = 500) -> pd.DataFrame:
        n = x.shape[0]

        x_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=x.device)
        x_b = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], device=x.device)

        history = {
            "step": [],
            "y_a": [],
            "y_b": [],
            "sign_gate": [],
            "separation": [],
        }

        step = 0

        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n):
                # Oja update
                datapt = x[perm[i]].unsqueeze(0)
                y = self(datapt)
                emb = self.input_to_embedding(datapt)
                dw = y * (emb - y * self.hidden.weight)

                with torch.no_grad():
                    self.embedding_to_output.weight += self.lr * dw

                # track statistics

                y_a = self(x_a).item()
                y_b = self(x_b).item()

                sign_gate = (y_a * y_b) < 0
                separation = abs(y_a - y_b)

                history["step"].append(step)
                history["y_a"].append(y_a)
                history["y_b"].append(y_b)
                history["sign_gate"].append(bool(sign_gate))
                history["separation"].append(separation)
                
                step += 1
        
        return pd.DataFrame(history)

class OjaDecayModel(nn.Module):
    def __init__(self, lr: float = 10e-3):
        super().__init__()

        self.lr = lr

        self.input_to_embedding = nn.Linear(5,16,bias=False)
        self.embedding_to_output = nn.Linear(16,1,bias=False)

        nn.init.uniform_(self.input_to_embedding.weight,a=-.01,b=.01)
        nn.init.uniform_(self.embedding_to_output.weight,a=-.01,b=.01)

        for param in self.parameters():
            param.requires_grad = False

    
    def forward(self, x: torch.Tensor):
        embedding = self.input_to_embedding(x)
        output = self.embedding_to_output(embedding)
        return output
    
    def learn_oja(self, x: torch.Tensor, epochs: int = 500) -> pd.DataFrame:
        n = x.shape[0]

        x_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=x.device)
        x_b = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], device=x.device)

        history = {
            "step": [],
            "y_a": [],
            "y_b": [],
            "sign_gate": [],
            "separation": [],
        }

        step = 0

        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n):
                # Oja update
                datapt = x[perm[i]].unsqueeze(0)
                y = self(datapt)
                emb = self.input_to_embedding(datapt)
                dw = y * (emb - y * self.hidden.weight)

                with torch.no_grad():
                    self.embedding_to_output.weight += self.lr * dw

                # track statistics

                y_a = self(x_a).item()
                y_b = self(x_b).item()

                sign_gate = (y_a * y_b) < 0
                separation = abs(y_a - y_b)

                history["step"].append(step)
                history["y_a"].append(y_a)
                history["y_b"].append(y_b)
                history["sign_gate"].append(bool(sign_gate))
                history["separation"].append(separation)
                
                step += 1
        
        return pd.DataFrame(history)