import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, input_dim, output_dim, activation):
        super(GraphConv, self).__init__()

        self.output_dim = output_dim
        self.activation = getattr(F, activation) if activation else None

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, inputs):
        X, A = inputs
        A_normalized = self._normalize(A)
        np.save('./pt_tensors/normalized', A_normalized.detach().cpu().numpy())
        x = torch.bmm(A_normalized, X)
        np.save('./pt_tensors/batch_matmult', x.detach().cpu().numpy())
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        np.save('./pt_tensors/gcnn_activation', x.detach().cpu().numpy())
        return x

    def _normalize(self, A, eps=1e-6):
        batch_size, n = A.shape[0], A.shape[1]
        A_hat = A.clone()
        A_hat.diagonal(dim1=1, dim2=2).fill_(0) # Remove self-loops
        A_hat += torch.eye(n, device=A.device).unsqueeze(0).expand(batch_size, -1, -1) # Add self-loops
        D_hat = torch.diag_embed(1. / (eps + A_hat.sum(dim=2).sqrt()))
        return torch.bmm(torch.bmm(D_hat, A_hat), D_hat) # Compute the normalized adjacency matrix


class FuncPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FuncPredictor, self).__init__()

        self.output_dim = output_dim
        self.output_layer = nn.Linear(input_dim, 2 * output_dim)
        torch.nn.init.ones_(self.output_layer.weight)

    def forward(self, x):
        x = self.output_layer(x)
        x = x.view(-1, self.output_dim, 2)
        return x # PyTorch's CrossEntropyLoss expects logits

class SumPooling(nn.Module):
    def __init__(self, axis):
        super(SumPooling, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.sum(dim=self.axis)
