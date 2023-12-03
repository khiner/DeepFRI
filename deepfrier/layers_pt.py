import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, input_dim, output_dim, activation):
        super(GraphConv, self).__init__()

        self.output_dim = output_dim
        self.activation = getattr(F, activation) if activation else None

        # Initialize the weights
        self.kernel = nn.Parameter(torch.Tensor(input_dim, self.output_dim))
        self.reset_parameters()

    def forward(self, inputs):
        X, A = inputs
        A_normalized = self._normalize(A)
        output = torch.bmm(A_normalized, X)
        output = torch.matmul(output, self.kernel)
        if self.activation:
            output = self.activation(output)
        return output

    def _normalize(self, A):
        batch_size, n = A.shape[0], A.shape[1]
        A_hat = A.clone()
        A_hat.diagonal(dim1=1, dim2=2).fill_(0) # Remove self-loops
        A_hat += torch.eye(n, device=A.device).unsqueeze(0).expand(batch_size, -1, -1) # Add self-loops
        D_hat = torch.diag_embed(torch.sum(A_hat, dim=2).pow(-0.5)) # Compute the degree matrix
        return D_hat @ A_hat @ D_hat # Compute the normalized adjacency matrix

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)

class FuncPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FuncPredictor, self).__init__()

        self.output_dim = output_dim
        self.output_layer = nn.Linear(input_dim, 2 * output_dim)

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
