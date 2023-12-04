import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .layers_pt import FuncPredictor, SumPooling, GraphConv

class DeepFRI(nn.Module):
    """ Class containing the GCN for predicting protein function. """
    def __init__(self, output_dim, n_channels, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4, model_name_prefix=None):
        super(DeepFRI, self).__init__()
        self.model_name_prefix = model_name_prefix

        gcnn_layers = []
        input_dim = n_channels
        for gc_dim in gc_dims:
            gc_layer = GraphConv(input_dim, gc_dim, activation='relu')
            gcnn_layers.append(gc_layer)
            input_dim = gc_dim
        self.gcnn_layers = nn.ModuleList(gcnn_layers)

        self.sum_pooling = SumPooling(axis=1)

        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(drop))
            input_dim = fc_dim
        self.fc_layers = nn.Sequential(*fc_layers)

        self.output_layer = FuncPredictor(input_dim, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.95, 0.99), weight_decay=l2_reg)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def forward(self, input_cmap, input_seq):
        x = input_seq
        for gc_layer in self.gcnn_layers:
            x = gc_layer([x, input_cmap])
        x = self.sum_pooling(x)
        x = self.fc_layers(x)
        out = self.output_layer(x)
        return out

    def train_model(self, device, train_loader, valid_loader, epochs=100):
        for epoch in range(epochs):
            self.train()
            total_loss, correct_predictions, total_predictions = 0.0, 0, 0
            for i, (input_cmap, input_seq, labels) in enumerate(train_loader):
                input_cmap, input_seq, labels = input_cmap.to(device), input_seq.to(device), labels.to(device)
                labels = torch.max(labels, 1)[1]  # Convert one-hot to class indices

                self.optimizer.zero_grad()
                outputs = self(input_cmap, input_seq)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Predicted class indices
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i} / {len(train_loader)}], Loss: {loss.item()}')

            loss = total_loss / len(train_loader)
            accurary = 100 * correct_predictions / total_predictions
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss}, Accuracy: {accurary}%')
            self.history['loss'].append(loss)
            self.history['acc'].append(accurary)

            # Validation
            self.eval()
            with torch.no_grad():
                total_loss, correct_predictions, total_predictions = 0.0, 0, 0
                for input_cmap, input_seq, labels in valid_loader:
                    input_cmap, input_seq, labels = input_cmap.to(device), input_seq.to(device), labels.to(device)
                    labels = torch.max(labels, 1)[1] # Convert one-hot to class indices

                    outputs = self(input_cmap, input_seq)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs, 1) # Predicted class indices
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)

                val_loss = total_loss / len(valid_loader)
                val_accuracy = 100 * correct_predictions / total_predictions
                print(f'Validation - Epoch [{epoch + 1}/{epochs}]: Loss: {val_loss}, Accuracy: {val_accuracy}%')
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_accuracy)

            # Save model checkpoint
            if self.model_name_prefix:
                torch.save(self.state_dict(), f'{self.model_name_prefix}_epoch_{epoch}.pth')

    def predict(self, input_cmap, input_seq):
        self.eval()
        with torch.no_grad():
            output = self(input_cmap, input_seq)
            return output.numpy()[0][:, 0]

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
