import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.onnx
from tqdm import tqdm

from torch_geometric.nn import GATConv
import torch_geometric.utils as pyg_utils

class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, input_dim, output_dim, activation):
        super(GraphConv, self).__init__()

        self.output_dim = output_dim
        self.activation = getattr(F, activation) if activation else None

        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs):
        X, A = inputs
        A_normalized = self._normalize(A)
        x = torch.bmm(A_normalized, X)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x

    def _normalize(self, A, eps=1e-6):
        batch_size, n = A.shape[0], A.shape[1]
        A_hat = A.clone()
        A_hat.diagonal(dim1=1, dim2=2).fill_(0) # Remove self-loops
        A_hat += torch.eye(n, device=A.device).unsqueeze(0).expand(batch_size, -1, -1) # Add self-loops
        D_hat = torch.diag_embed(1. / (eps + A_hat.sum(dim=2).sqrt()))
        return torch.bmm(torch.bmm(D_hat, A_hat), D_hat) # Compute the normalized adjacency matrix

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.6, concat=True, use_bias=True, activation=F.relu):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=num_heads, dropout=dropout, concat=concat, bias=use_bias)
        self.activation = activation

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        if self.activation:
            x = self.activation(x)
        return x

class DeepFRI(nn.Module):
    """GCN model for predicting protein function."""
    def __init__(self, output_dim, n_channels, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4, model_name_prefix=None):
        super(DeepFRI, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.l2_reg = l2_reg

        num_heads = 4
        # Encoding layer
        lm_dim = 1024
        self.input_layer = nn.Sequential(
            nn.Linear(n_channels, lm_dim, bias=False),
            nn.ReLU(),
            nn.Linear(lm_dim, gc_dims[0] * num_heads, bias=False)
        )
        self.gat_layers = nn.ModuleList(
            [GATLayer(in_channels=(gc_dims[i - 1] * num_heads if i > 0 else gc_dims[0] * num_heads),
                      out_channels=gc_dim,
                      num_heads=num_heads,
                      dropout=drop,
                      concat=True,
                      use_bias=True,
                      activation='relu') for i, gc_dim in enumerate(gc_dims)])

        final_dim = gc_dims[-1] * num_heads if gc_dims else n_channels
    
        # Fully connected layers
        fc_layers = []
        for l, fc_dim in enumerate(fc_dims):
            fc_layers.append(nn.Linear(final_dim if l == 0 else fc_dims[l-1], fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout((l + 1) * drop))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.output_layer = nn.Linear(fc_dims[-1] if fc_dims else final_dim, output_dim) # PyTorch's BCEWithLogitsLoss expects logits

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.95, 0.99), weight_decay=self.l2_reg)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def forward(self, input_cmap, input_seq):
        x = self.input_layer(input_seq)
        edge_index, _ = pyg_utils.dense_to_sparse(input_cmap)  # Convert adjacency matrix to edge indices

        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)

        x = x.sum(dim=1)  # Sum over nodes
        x = self.fc_layers(x)
        out = self.output_layer(x)
        return out

    def process_batch(self, batch, device):
        cmap, seq, labels = batch
        cmap, seq, labels = cmap.to(device), seq.to(device), labels.to(device)

        outputs = self(cmap, seq)
        loss = self.criterion(outputs, labels)
        return outputs, labels, loss

    def calculate_accuracy(self, outputs, labels):
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct_predictions = (predictions == labels).sum().item()
        total_predictions = labels.numel()
        return correct_predictions, total_predictions

    def run_epoch(self, epoch, total_epochs, loader, device, is_training):
        if is_training:
            self.train()
        else:
            self.eval()

        total_loss, correct_predictions, total_predictions = 0.0, 0, 0
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        for i, batch in progress_bar:
            if is_training:
                self.optimizer.zero_grad()

            outputs, labels, loss = self.process_batch(batch, device)

            if is_training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            correct, total = self.calculate_accuracy(outputs, labels)
            correct_predictions += correct
            total_predictions += total

            accuracy = 100 * correct_predictions / total_predictions
            progress_bar.set_description(f'Epoch [{epoch}/{total_epochs}]: Loss: {loss.item():.3f}, Acc: {accuracy:.3f}')

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct_predictions / total_predictions
        return avg_loss, accuracy

    def fit(self, device, train_loader, valid_loader, epochs=100):
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self.run_epoch(epoch, epochs, train_loader, device, is_training=True)
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss}, Accuracy: {train_accuracy}%')
            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_accuracy)

            val_loss, val_accuracy = self.run_epoch(epoch, epochs, valid_loader, device, is_training=False)
            print(f'Validation - Epoch [{epoch}/{epochs}]: Loss: {val_loss}, Accuracy: {val_accuracy}%')
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_accuracy)

            self.save_model(f'epoch_{epoch}')  # Save checkpoint

    def predict(self, input_cmap, input_seq):
        self.eval()
        with torch.no_grad():
            output = self(input_cmap, input_seq)
            return output.numpy()[0][:, 0]

    def save_onnx(self, train_loader):
        input_cmap, input_seq, _ = next(iter(train_loader))
        torch.onnx.export(self, (input_cmap, input_seq), './model.onnx', opset_version=11)

    def save_model(self, file_stem_suffix):
        prefix = f'{self.model_name_prefix}_' if self.model_name_prefix else ''
        file_path = f'checkpoints/{prefix}{file_stem_suffix}.pth'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

    def plot_losses(self):
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(self.history['loss'], '-')
        plt.plot(self.history['val_loss'], '-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_name_prefix}_model_loss.png', bbox_inches='tight')

        plt.figure()
        plt.plot(self.history['acc'], '-')
        plt.plot(self.history['val_acc'], '-')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_name_prefix}_model_accuracy.png', bbox_inches='tight')
