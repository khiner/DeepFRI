import csv
import json
import os
import pickle
import argparse

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdata.datapipes.iter import FileLister, FileOpener


from deepfrier.DeepFRI_pt import DeepFRI
from deepfrier.utils import seq2onehot
from deepfrier.utils import load_GO_annot, load_EC_annot

def parse_tfrecord(serialized, n_goterms, channels, cmap_type, cmap_thresh, ont):
    features = {
        f'{cmap_type}_dist_matrix': tf.io.VarLenFeature(dtype=tf.float32),
        'seq_1hot': tf.io.VarLenFeature(dtype=tf.float32),
        f'{ont}_labels': tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        'L': tf.io.FixedLenFeature([1], dtype=tf.int64)
    }

    parsed_example = serialized
    L = parsed_example['L'][0].numpy()

    A = parsed_example[f'{cmap_type}_dist_matrix'].to_dense().numpy()
    A = A.reshape(L, L)
    A_cmap = (A <= cmap_thresh).astype(np.float32)

    S = parsed_example['seq_1hot'].to_dense().numpy()
    S = S.reshape(L, channels)

    labels = parsed_example[f'{ont}_labels'].numpy().astype(np.float32)
    inverse_labels = (labels == 0).astype(np.float32)
    y = np.stack([labels, inverse_labels], axis=-1)

    return torch.tensor(A_cmap, dtype=torch.float32), torch.tensor(S, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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
    length = 0

    datapipe1 = FileLister(filenames, "*.tfrecords")
    datapipe2 = FileOpener(datapipe1, mode="b")

    for _ in datapipe2:
        length += 1
    
    tfrecord_loader_dp = datapipe2.load_from_tfrecord(length=length)
    tfrecord_loader_dp = tfrecord_loader_dp.map(lambda x: parse_tfrecord(x, n_goterms, channels, cmap_type, cmap_thresh, ont))

    return tfrecord_loader_dp


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gcd', '--gc_dims', type=int, default=[128, 128, 256], nargs='+', help="Dimensions of GraphConv layers.")
    parser.add_argument('-fcd', '--fc_dims', type=int, default=[], nargs='+', help="Dimensions of fully connected layers (after GraphConv layers).")
    parser.add_argument('-drop', '--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('-l2', '--l2_reg', type=float, default=1e-4, help="L2 regularization coefficient.")
    parser.add_argument('-lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('-pd', '--pad_len', type=int, help="Pad length (max len of protein sequences in train set).")
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument('--cmap_type', type=str, default='ca', choices=['ca', 'cb'], help="Contact maps type.")
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help="Distance cutoff for thresholding contact maps.")
    parser.add_argument('--model_name', type=str, default='GCN-PDB_MF', help="Name of the GCN model.")
    parser.add_argument('--train_tfrecord_fn', type=str, default="./preprocessing/data/downloaded/PDB-GO/PDB_GO_train", help="Train tfrecords.")
    parser.add_argument('--valid_tfrecord_fn', type=str, default="./preprocessing/data/downloaded/PDB-GO/PDB_GO_valid", help="Valid tfrecords.")
    parser.add_argument('--annot_fn', type=str, default="./preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv", help="File (*tsv) with GO term annotations.")
    parser.add_argument('--test_list', type=str, default="./preprocessing/data/nrPDB-GO_2019.06.18_test.csv", help="File with test PDB chains.")

    args = parser.parse_args()
    print("### Args:")
    print(args)

    # Load annotations
    annot_load_fn = load_EC_annot if args.ontology == 'ec' else load_GO_annot
    prot2annot, goterms, gonames, counts = annot_load_fn(args.annot_fn)
    goterms, gonames = goterms[args.ontology], gonames[args.ontology]
    output_dim = len(goterms)

    batch_size, n_channels = args.batch_size, 26
    pad_len, cmap_type, cmap_thresh, ont = args.pad_len, args.cmap_type, args.cmap_thresh, args.ontology

    train_datapipe = get_dataset(f'{args.train_tfrecord_fn}', output_dim, n_channels, cmap_type, cmap_thresh, ont)
    valid_datapipe = get_dataset(f'{args.valid_tfrecord_fn}', output_dim, n_channels, cmap_type, cmap_thresh, ont)

    train_loader = DataLoader(dataset=train_datapipe, batch_size=batch_size, shuffle=True, collate_fn=pad_batch, num_workers=4)
    valid_loader = DataLoader(dataset=valid_datapipe, batch_size=batch_size, shuffle=False, collate_fn=pad_batch, num_workers=4)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    print(f'### Training model: {args.model_name} on {output_dim} GO terms.')
    model = DeepFRI(output_dim=output_dim, n_channels=n_channels, gc_dims=args.gc_dims, fc_dims=args.fc_dims,
                    lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, model_name_prefix=args.model_name).to(device)
    model.train_model(device, train_loader, valid_loader, epochs=args.epochs)
    model.save_model()
    model.plot_losses()

    # Save model params to json
    with open(f'{args.model_name}_model_params.json', 'w') as fw:
        out_params = vars(args)
        out_params['goterms'] = goterms
        out_params['gonames'] = gonames
        json.dump(out_params, fw, indent=1)

    # Use trained model for prediction
    Y_pred, Y_true = [], []
    proteins = []
    cmaps_path = './examples/pdb_cmaps'
    with open(args.test_list, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # header
        for row in csv_reader:
            protein = row[0]
            cmap_path = f'{cmaps_path}/{protein}.npz'
            if not os.path.isfile(cmap_path):
                print(f'### Skipping {protein}: No cmap found.')
                continue

            cmap = np.load(cmap_path)
            sequence = str(cmap['seqres'])
            Ca_dist = cmap['C_alpha']

            A = np.double(Ca_dist < args.cmap_thresh)
            S = seq2onehot(sequence)

            S = S.reshape(1, *S.shape)
            A = A.reshape(1, *A.shape)

            # Results
            proteins.append(protein)
            Y_pred.append(model.predict([A, S]).reshape(1, output_dim))
            Y_true.append(prot2annot[protein][args.ontology].reshape(1, output_dim))

    pickle.dump(
        {
            'proteins': np.asarray(proteins),
            'Y_pred': np.concatenate(Y_pred, axis=0),
            'Y_true': np.concatenate(Y_true, axis=0),
            'ontology': args.ontology,
            'goterms': goterms,
            'gonames': gonames,
        },
        open(f'{args.model_name}_results.pckl', 'wb')
    )
