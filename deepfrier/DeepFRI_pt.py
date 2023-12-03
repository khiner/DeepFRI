import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .layers_pt import FuncPredictor, SumPooling, GraphConv

def parse_tfrecord(serialized, n_goterms, channels, cmap_type, cmap_thresh, ont):
    features = {
        f'{cmap_type}_dist_matrix': tf.io.VarLenFeature(dtype=tf.float32),
        'seq_1hot': tf.io.VarLenFeature(dtype=tf.float32),
        f'{ont}_labels': tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        'L': tf.io.FixedLenFeature([1], dtype=tf.int64)
    }

    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
    L = parsed_example['L'][0].numpy()

    A = tf.sparse.to_dense(parsed_example[f'{cmap_type}_dist_matrix']).numpy()
    A = A.reshape(L, L)
    A_cmap = (A <= cmap_thresh).astype(np.float32)

    S = tf.sparse.to_dense(parsed_example['seq_1hot']).numpy()
    S = S.reshape(L, channels)

    labels = parsed_example[f'{ont}_labels'].numpy().astype(np.float32)
    inverse_labels = (labels == 0).astype(np.float32)
    y = np.stack([labels, inverse_labels], axis=-1)

    return A_cmap, S, y

def pad_tensors(tensors):
    """Pad a list of tensors to the same size in the first two dimensions. """
    max_size = max(tensor.shape[0] for tensor in tensors), max(tensor.shape[1] for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        padded_tensor = torch.full(max_size, 0, dtype=tensor.dtype)
        padded_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)

def pad_batch(batch):
    cmaps, seqs, labels = zip(*batch)
    cmaps_padded = pad_tensors([cmap for cmap in cmaps])
    seqs_padded = pad_sequence([seq for seq in seqs], batch_first=True, padding_value=0)
    labels = torch.stack([label for label in labels])

    return cmaps_padded, seqs_padded, labels

class TFRecordDataset(Dataset):
    def __init__(self, filenames, n_goterms, channels, cmap_type, cmap_thresh, ont):
        self.filenames = tf.io.gfile.glob(filenames)
        self.n_goterms = n_goterms
        self.channels = channels
        self.cmap_type = cmap_type
        self.cmap_thresh = cmap_thresh
        self.ont = ont
        self.indexes = self._create_indexes()

    def _create_indexes(self):
        indexes = []
        total_count = 0
        for filename in self.filenames:
            count = sum(1 for _ in tf.data.TFRecordDataset(filename))
            indexes.append((filename, total_count, total_count + count))
            total_count += count
        return indexes

    def _get_record(self, global_idx):
        for filename, start_idx, end_idx in self.indexes:
            if start_idx <= global_idx < end_idx:
                local_idx = global_idx - start_idx
                record = next(iter(tf.data.TFRecordDataset(filename).skip(local_idx).take(1)))
                return parse_tfrecord(record, self.n_goterms, self.channels, self.cmap_type, self.cmap_thresh, self.ont)
        raise IndexError('Index out of dataset range')

    def __len__(self):
        return self.indexes[-1][-1] if self.indexes else 0

    def __getitem__(self, idx):
        cmap, seq, label = self._get_record(idx)
        return torch.tensor(cmap, dtype=torch.float32), torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def get_dataset(filenames, n_goterms=347, channels=26, cmap_type='ca', cmap_thresh=10.0, ont='mf'):
    return TFRecordDataset(filenames, n_goterms, channels, cmap_type, cmap_thresh, ont)

class DeepFRI(nn.Module):
    """ Class containing the GCN for predicting protein function. """
    def __init__(self, output_dim, n_channels=26, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4, model_name_prefix=None):
        super(DeepFRI, self).__init__()
        self.output_dim = output_dim
        self.n_channels = n_channels
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

    def forward(self, input_cmap, input_seq):
        x = input_seq
        for gc_layer in self.gcnn_layers:
            x = gc_layer([x, input_cmap])
        x = self.sum_pooling(x)
        x = self.fc_layers(x)
        out = self.output_layer(x)
        return out

    def train_model(self, device, train_tfrecord_fn, valid_tfrecord_fn, epochs=100, batch_size=64, pad_len=1200, cmap_type='ca', cmap_thresh=10.0, ont='mf'):
        train_dataset = get_dataset(train_tfrecord_fn, self.output_dim, self.n_channels, cmap_type, cmap_thresh, ont)
        valid_dataset = get_dataset(valid_tfrecord_fn, self.output_dim, self.n_channels, cmap_type, cmap_thresh, ont)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for i, (input_cmap, input_seq, labels) in enumerate(train_loader):
                input_cmap, input_seq, labels = input_cmap.to(device), input_seq.to(device), labels.to(device)
                labels = torch.max(labels, 1)[1]  # Convert one-hot to class indices

                self.optimizer.zero_grad()
                outputs = self(input_cmap, input_seq)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader)}')

            # Validation
            self.eval()
            with torch.no_grad():
                validation_loss, correct_predictions, total_predictions = 0.0, 0, 0
                for input_cmap, input_seq, labels in valid_loader:
                    input_cmap, input_seq, labels = input_cmap.to(device), input_seq.to(device), labels.to(device)
                    labels = torch.max(labels, 1)[1] # Convert one-hot to class indices

                    outputs = self(input_cmap, input_seq)
                    loss = self.criterion(outputs, labels)
                    validation_loss += loss.item()

                    _, predicted = torch.max(outputs, 1) # Get the predicted class indices
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)

                avg_validation_loss = validation_loss / len(valid_loader)
                validation_accuracy = 100 * correct_predictions / total_predictions

                print(f'Validation - Epoch [{epoch + 1}/{epochs}]: Average Loss: {avg_validation_loss}, Accuracy: {validation_accuracy}%')

            # Save model checkpoint
            if self.model_name_prefix:
                torch.save(self.state_dict(), f'{self.model_name_prefix}_epoch_{epoch}.pth')

    def predict(self, input_cmap, input_seq):
        self.eval()
        with torch.no_grad():
            output = self(input_cmap, input_seq)
            return output.numpy()[0][:, 0]

    # TODO - implement loss/acc history
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
