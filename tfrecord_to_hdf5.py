import argparse
from glob import glob
import h5py
import tensorflow as tf
import numpy as np

def collect_metadata(tfrecord_filenames):
    metadata = {}
    total_records = 0
    for tfrecord_filename in tfrecord_filenames:
        for record in tf.data.TFRecordDataset(tfrecord_filename):
            total_records += 1
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            for key, feature in example.features.feature.items():
                if feature.HasField('bytes_list'):
                    dtype = 'string'
                    feature_len = None  # Variable-length strings
                elif feature.HasField('float_list'):
                    dtype = 'float_list'
                    feature_len = len(feature.float_list.value)
                elif feature.HasField('int64_list'):
                    dtype = 'int64_list'
                    feature_len = len(feature.int64_list.value)

                # Update metadata with the max length encountered
                if key not in metadata or (metadata[key][1] is not None and feature_len > metadata[key][1]):
                    metadata[key] = (dtype, feature_len)

    return metadata, total_records

def tfrecord_to_hdf5(tfrecord_filenames, output_filename, metadata, total_records):
    with h5py.File(output_filename, 'w') as hdf5_file:
        # Create datasets based on metadata
        datasets = {}
        for key, (dtype, _) in metadata.items():
            if dtype == 'string':
                dt = h5py.special_dtype(vlen=str)
                datasets[key] = hdf5_file.create_dataset(key, (total_records,), dtype=dt)
            elif dtype in ['float_list', 'int64_list']:
                dt = h5py.special_dtype(vlen=np.float32 if dtype == 'float_list' else np.int64)
                datasets[key] = hdf5_file.create_dataset(key, (total_records,), dtype=dt)

        # Write data to HDF5 file
        index = 0
        for tfrecord_filename in tfrecord_filenames:
            for record in tf.data.TFRecordDataset(tfrecord_filename):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())

                for key, feature in example.features.feature.items():
                    dataset = datasets[key]
                    if feature.HasField('bytes_list'):
                        string_data = [bytes.decode('utf-8') for bytes in feature.bytes_list.value]
                        dataset[index] = string_data
                    elif feature.HasField('float_list'):
                        data = np.array(feature.float_list.value, dtype=np.float32)
                        dataset[index] = data
                    elif feature.HasField('int64_list'):
                        data = np.array(feature.int64_list.value, dtype=np.int64)
                        dataset[index] = data

                index += 1

##### TESTING #####

def parse_hdf5_record(filename, i, cmap_type, cmap_thresh, ont, channels):
    with h5py.File(filename, 'r') as hdf5_file:
        L = hdf5_file['L'][i][0]

        A = hdf5_file[f'{cmap_type}_dist_matrix'][i]
        A = A.reshape(L, L)
        A_cmap = (A <= cmap_thresh).astype(np.float32)

        S = hdf5_file['seq_1hot'][i]
        S = S.reshape(L, channels)

        labels = hdf5_file[f'{ont}_labels'][i]
        inverse_labels = (labels == 0).astype(np.float32)
        y = np.stack([labels, inverse_labels], axis=-1)

        return A_cmap, S, y

# This is exactly from `DeepFRI.py`, with the exception of the hard-coded `n_goterms` value.
def parse_tfrecord(serialized, cmap_type, cmap_thresh, ont,  channels):
    n_goterms = 489
    features = {
        cmap_type + '_dist_matrix': tf.io.VarLenFeature(dtype=tf.float32),
        "seq_1hot": tf.io.VarLenFeature(dtype=tf.float32),
        ont + "_labels": tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64)
    }

    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    L = parsed_example['L'][0]

    A_shape = tf.stack([L, L])
    A = parsed_example[cmap_type + '_dist_matrix']
    A = tf.cast(A, tf.float32)
    A = tf.sparse.to_dense(A)
    A = tf.reshape(A, A_shape)

    A_cmap = tf.cast(tf.less_equal(A, cmap_thresh), tf.float32)

    S_shape = tf.stack([L, channels])
    S = parsed_example['seq_1hot']
    S = tf.cast(S, tf.float32)
    S = tf.sparse.to_dense(S)
    S = tf.reshape(S, S_shape)

    labels = parsed_example[ont + '_labels']
    labels = tf.cast(labels, tf.float32)

    inverse_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32)  # [batch, classes]
    y = tf.stack([labels, inverse_labels], axis=-1)  # labels, inverse labels
    y = tf.reshape(y, shape=[n_goterms, 2])  # [batch, classes, Pos-Neg].

    return A_cmap, S, y

def compare_records(tfrecord_filename, hdf5_filename, num_records):
    tfrecords = [record for record in tf.data.TFRecordDataset(tfrecord_filename).take(num_records)]
    for i in range(num_records):
        tfrecord_record = tfrecords[i]
        tfrecord_A_cmap, tfrecord_S, tfrecord_y = parse_tfrecord(tfrecord_record, cmap_type='ca', cmap_thresh=10.0, ont='mf', channels=26)
        hdf5_A_cmap, hdf5_S, hdf5_y = parse_hdf5_record(hdf5_filename, i, cmap_type='ca', cmap_thresh=10.0, ont='mf', channels=26)
        if not np.array_equal(tfrecord_A_cmap, hdf5_A_cmap):
            return False
        if not np.array_equal(tfrecord_S, hdf5_S):
            return False
        if not np.array_equal(tfrecord_y, hdf5_y):
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='./preprocessing/data/downloaded/PDB-GO-TRAIN', help='Directory of TFRecord files to convert to HDF5.')
    parser.add_argument('--output', type=str, default='./preprocessing/data/downloaded/PDB-GO-TRAIN/pdb_go_train.hdf5', help='Output HDF5 file.')

    args = parser.parse_args()

    tfrecord_filenames = glob(f'{args.dir}/*.tfrecords')
    tfrecord_filenames.sort()
    metadata, total_records = collect_metadata(tfrecord_filenames)
    tfrecord_to_hdf5(tfrecord_filenames, args.output, metadata, total_records)

    # Test that the records are identical.
    # tfrecord_filename = './preprocessing/data/downloaded/PDB-GO-VALID/PDB_GO_valid_00-of-03.tfrecords'
    # hdf5_filename = './preprocessing/data/downloaded/PDB-GO-VALID/pdb_go_valid.hdf5'
    # compare_num_records = 500
    # if compare_records(tfrecord_filename, hdf5_filename, compare_num_records):
    #     print("Records are identical.")
    # else:
    #     print("Records differ.")
