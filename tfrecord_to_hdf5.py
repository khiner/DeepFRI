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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='./preprocessing/data/downloaded/PDB-GO-TRAIN', help='Directory of TFRecord files to convert to HDF5.')
    parser.add_argument('--output', type=str, default='./preprocessing/data/downloaded/PDB-GO-TRAIN/pdb_go_train.hdf5', help='Output HDF5 file.')

    args = parser.parse_args()

    tfrecord_filenames = glob(f'{args.dir}/*.tfrecords')
    metadata, total_records = collect_metadata(tfrecord_filenames)
    tfrecord_to_hdf5(tfrecord_filenames, args.output, metadata, total_records)
