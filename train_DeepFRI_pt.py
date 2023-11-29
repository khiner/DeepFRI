import csv
import json
import os
import pickle
import torch

import argparse
import numpy as np

from deepfrier.DeepFRI_pt import DeepFRI
from deepfrier.utils import seq2onehot
from deepfrier.utils import load_GO_annot, load_EC_annot


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

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    print(f'### Training model: {args.model_name} on {output_dim} GO terms.')
    model = DeepFRI(output_dim=output_dim, n_channels=26, gc_dims=args.gc_dims, fc_dims=args.fc_dims,
                    lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, model_name_prefix=args.model_name).to(device)
    model.train_model(device, f'{args.train_tfrecord_fn}*', f'{args.valid_tfrecord_fn}*', epochs=args.epochs, batch_size=args.batch_size, pad_len=args.pad_len,
                cmap_type=args.cmap_type, cmap_thresh=args.cmap_thresh, ont=args.ontology)
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
