import glob
import tensorflow as tf

from .layers import FuncPredictor, SumPooling, GraphConv

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def _parse_function_gcn(serialized, n_goterms, channels=26, cmap_type='ca', cmap_thresh=10.0, ont='mf'):
    features = {
        cmap_type + '_dist_matrix': tf.io.VarLenFeature(dtype=tf.float32),
        "seq_1hot": tf.io.VarLenFeature(dtype=tf.float32),
        ont + "_labels": tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64)
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L'][0]

    A_shape = tf.stack([L, L])
    A = parsed_example[cmap_type + '_dist_matrix']
    A = tf.cast(A, tf.float32)
    A = tf.sparse.to_dense(A)
    A = tf.reshape(A, A_shape)

    # threshold distances
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

    return {'cmap': A_cmap, 'seq': S}, y


def get_batched_dataset(filenames, batch_size=64, pad_len=1000, n_goterms=347, channels=26, cmap_type='ca', cmap_thresh=10.0, ont='mf'):
    # settings to read from all the shards in parallel
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # list all files
    filenames = tf.io.gfile.glob(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)

    # Parse the serialized data in the TFRecords files.
    dataset = dataset.map(lambda x: _parse_function_gcn(x, n_goterms=n_goterms, channels=channels, cmap_type=cmap_type, cmap_thresh=cmap_thresh, ont=ont))

    # Randomizes input using a window of 2000 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=2000 + 3*batch_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=({'cmap': [pad_len, pad_len], 'seq': [pad_len, channels]}, [None, 2]))
    dataset = dataset.repeat()

    return dataset


class DeepFRI(object):
    """ Class containig the GCN for predicting protein function. """
    def __init__(self, output_dim, n_channels=26, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4, model_name_prefix=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        :param gc_dims: {list <int>} number of hidden units in GConv layers
        :param fc_dims: {list <int>} number of hiddne units in Dense layers
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :model_name_prefix: {string} name of a deepFRI model to be saved
        """
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.model_name_prefix = model_name_prefix

        # Build and compile model
        self.gc_layer = 'GraphConv' # Simplifying to use only one type of GC layer for now.
        print ("### Compiling DeepFRI model with %s layer..." % (self.gc_layer))

        input_cmap = tf.keras.layers.Input(shape=(None, None), name='cmap')
        input_seq = tf.keras.layers.Input(shape=(None, n_channels), name='seq')

        # Encoding layers
        lm_dim = 1024
        x_aa = tf.keras.layers.Dense(lm_dim, use_bias=False, name='AA_embedding')(input_seq)
        x = tf.keras.layers.Activation('relu')(x_aa)

        # Graph convolution layers
        gcnn_layers = []
        for l in range(len(gc_dims)):
            x = GraphConv(
                gc_dims[l], activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name=f'{self.gc_layer}_{l+1}'
            )([x, input_cmap])
            gcnn_layers.append(x)

        x = tf.keras.layers.Concatenate(name='GCNN_concatenate')(gcnn_layers) if len(gcnn_layers) > 1 else gcnn_layers[-1]
        x = SumPooling(axis=1, name='SumPooling')(x)

        # Dense layers
        for l in range(len(fc_dims)):
            x = tf.keras.layers.Dense(units=fc_dims[l], activation='relu')(x)
            x = tf.keras.layers.Dropout((l + 1)*drop)(x)

        output_layer = FuncPredictor(output_dim=output_dim, name='labels')(x)

        self.model = tf.keras.Model(inputs=[input_cmap, input_seq], outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.99)
        pred_loss = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(optimizer=optimizer, loss=pred_loss, metrics=['acc'])
        print (self.model.summary())

    def train(self, train_tfrecord_fn, valid_tfrecord_fn,
              epochs=100, batch_size=64, pad_len=1200, cmap_type='ca', cmap_thresh=10.0, ont='mf'):

        n_train_records = sum(1 for f in glob.glob(train_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        n_valid_records = sum(1 for f in glob.glob(valid_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        print ("### Training on: ", n_train_records, "contact maps.")
        print ("### Validating on: ", n_valid_records, "contact maps.")

        # train tfrecords
        batch_train = get_batched_dataset(train_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          cmap_type=cmap_type,
                                          cmap_thresh=cmap_thresh,
                                          ont=ont)

        # validation tfrecords
        batch_valid = get_batched_dataset(valid_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          cmap_type=cmap_type,
                                          cmap_thresh=cmap_thresh,
                                          ont=ont)

        # early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # model checkpoint
        mc = tf.keras.callbacks.ModelCheckpoint(self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1,
                                                save_best_only=True, save_weights_only=True)

        # fit model
        history = self.model.fit(
            batch_train,
            epochs=epochs,
            validation_data=batch_valid,
            steps_per_epoch=n_train_records//batch_size,
            validation_steps=n_valid_records//batch_size,
            callbacks=[es, mc])

        self.history = history.history

    def predict(self, input_data):
        return self.model(input_data).numpy()[0][:, 0]

    def plot_losses(self):
        plt.figure()
        plt.plot(self.history['loss'], '-')
        plt.plot(self.history['val_loss'], '-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_loss.png', bbox_inches='tight')

        plt.figure()
        plt.plot(self.history['acc'], '-')
        plt.plot(self.history['val_acc'], '-')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')

    def save_model(self):
        self.model.save(self.model_name_prefix + '.hdf5')

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_name_prefix + '.hdf5',
                                                custom_objects={self.gc_layer: GraphConv,
                                                                'FuncPredictor': FuncPredictor,
                                                                'SumPooling': SumPooling
                                                                })
