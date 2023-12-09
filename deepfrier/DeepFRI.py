import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.onnx

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

class DeepFRI(nn.Module):
    """GCN model for predicting protein function."""
    def __init__(self, output_dim, n_channels, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4, model_name_prefix=None):
        super(DeepFRI, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.l2_reg = l2_reg

        # Encoding layer
        lm_dim = 1024
        self.input_layer = nn.Sequential(
            nn.Linear(n_channels, lm_dim, bias=False),
            nn.ReLU()
        )

        gcnn_layers = []
        input_dim = lm_dim
        for gc_dim in gc_dims:
            gc_layer = GraphConv(input_dim, gc_dim, activation='relu')
            gcnn_layers.append(gc_layer)
            input_dim = gc_dim
        self.gcnn_layers = nn.ModuleList(gcnn_layers)
        input_dim = sum(gc_dims) # GCNN layers are concatenated

        fc_layers = []
        for l, fc_dim in enumerate(fc_dims):
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout((l + 1) * drop))
            input_dim = fc_dim
        self.fc_layers = nn.Sequential(*fc_layers)

        self.output_layer = nn.Linear(input_dim, output_dim) # PyTorch's BCEWithLogitsLoss expects logits

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.95, 0.99), weight_decay=self.l2_reg)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def forward(self, input_cmap, input_seq):
        x = self.input_layer(input_seq)
        gcnn_outputs = []
        for gc_layer in self.gcnn_layers:
            x = gc_layer([x, input_cmap])
            gcnn_outputs.append(x)

        # Concatenate along the feature dimension
        x = torch.cat(gcnn_outputs, dim=2) if len(gcnn_outputs) > 1 else gcnn_outputs[0]
        x = x.sum(dim=1)
        x = self.fc_layers(x)
        out = self.output_layer(x)
        return out

    def train_model(self, device, train_loader, valid_loader, epochs=100):
        for epoch in range(epochs):
            self.train()
            total_loss, correct_predictions, total_predictions = 0.0, 0, 0
            for i, (cmap, seq, labels) in enumerate(train_loader):
                cmap, seq, labels = cmap.to(device), seq.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self(cmap, seq)
                loss = self.criterion(outputs, labels)
                # Optional L2 regularization on only the GCNN layers
                # for gcnn_layer in self.gcnn_layers:
                    # loss += self.l2_reg * torch.norm(gcnn_layer.linear.weight, p=2)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Caluclate multi-label prediction accuracy
                predictions = (torch.sigmoid(outputs) >= 0.5).float() # Convert logits to binary predictions with a 0.5 threshold
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.shape[0] * labels.shape[1]

                accuracy = torch.mean((predictions == labels).float())
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i} / {len(train_loader)}], Loss: {loss.item()}, Acc: {accuracy}')

            loss = total_loss / len(train_loader)
            accurary = 100 * correct_predictions / total_predictions
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss}, Accuracy: {accurary}%')
            self.history['loss'].append(loss)
            self.history['acc'].append(accurary)

            # Validation
            self.eval()
            with torch.no_grad():
                total_loss, correct_predictions, total_predictions = 0.0, 0, 0
                for cmap, seq, labels in valid_loader:
                    cmap, seq, labels = cmap.to(device), seq.to(device), labels.to(device)

                    outputs = self(cmap, seq)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                    predictions = (torch.sigmoid(outputs) >= 0.5).float() # Convert logits to binary predictions with a 0.5 threshold
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.shape[0] * labels.shape[1]

                val_loss = total_loss / len(valid_loader)
                val_accuracy = 100 * correct_predictions / total_predictions
                print(f'Validation - Epoch [{epoch + 1}/{epochs}]: Loss: {val_loss}, Accuracy: {val_accuracy}%')
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_accuracy)

            self.save_model(f'epoch_{epoch}') # Save checkpoint

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
