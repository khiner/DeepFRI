import json
import argparse

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary

from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import load_GO_annot, load_EC_annot

class HDF5Dataset(Dataset):
    def __init__(self, filename, cmap_type, cmap_thresh, ont, channels):
        self.filename = filename
        self.cmap_type = cmap_type
        self.cmap_thresh = cmap_thresh
        self.ont = ont
        self.channels = channels
        self.hdf5_file = None
        with h5py.File(self.filename, 'r') as hdf5_file:
            self.num_records = hdf5_file['L'].shape[0]

    def __getitem__(self, i):
        # Open the file if it is not already open
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.filename, 'r')

        L = self.hdf5_file['L'][i][0]

        A = self.hdf5_file[f'{self.cmap_type}_dist_matrix'][i]
        A = A.reshape(L, L)
        A_cmap = (A <= self.cmap_thresh).astype(np.float32)

        S = self.hdf5_file['seq_1hot'][i]
        S = S.reshape(L, self.channels)

        y = self.hdf5_file[f'{self.ont}_labels'][i]

        return torch.tensor(A_cmap, dtype=torch.float32), torch.tensor(S, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.num_records

    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()

    def __del__(self):
        self.close()

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
    return pad_tensors(cmaps), pad_sequence(seqs, batch_first=True), torch.stack(labels)


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
    parser.add_argument('--train_hdf5_file', type=str, default="./preprocessing/data/downloaded/PDB-GO-TRAIN/pdb_go_train.hdf5", help="Train HDF5 file.")
    parser.add_argument('--valid_hdf5_file', type=str, default="./preprocessing/data/downloaded/PDB-GO-VALID/pdb_go_valid.hdf5", help="Valid HDF5 file.")
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

    print('### Processing Data')
    train_dataset = HDF5Dataset(args.train_hdf5_file, cmap_type, cmap_thresh, ont, n_channels)
    valid_dataset = HDF5Dataset(args.valid_hdf5_file, cmap_type, cmap_thresh, ont, n_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch, num_workers=4, pin_memory=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    model = DeepFRI(output_dim=output_dim, n_channels=n_channels, gc_dims=args.gc_dims, fc_dims=args.fc_dims,
                    lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, model_name_prefix=args.model_name).to(device)
    print(f'### Training model: {args.model_name} on {output_dim} GO terms.')
    summary(model, device=device)
    model.fit(device, train_loader, valid_loader, epochs=args.epochs)
    # model.save_onnx(train_loader)
    model.save_model('final')
    model.plot_losses()

    # Save model params to json
    with open(f'{args.model_name}_model_params.json', 'w') as fw:
        out_params = vars(args)
        out_params['goterms'] = goterms
        out_params['gonames'] = gonames
        json.dump(out_params, fw, indent=1)
