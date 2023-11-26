import tensorflow as tf

class GraphConv(tf.keras.layers.Layer):
    """
         Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, output_dim, activation, kernel_regularizer=None, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            name='kernel',
            regularizer=self.kernel_regularizer,
            trainable=True)

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        D_hat = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A_hat, axis=2))))
        return tf.matmul(tf.matmul(D_hat, A_hat), D_hat)

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]), inputs[0])
        output = tf.keras.backend.dot(output, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

class FuncPredictor(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuncPredictor, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.output_layer = tf.keras.layers.Dense(2*output_dim)
        self.reshape = tf.keras.layers.Reshape(target_shape=(output_dim, 2))
        self.softmax = tf.keras.layers.Softmax(axis=-1, name='labels')

    def call(self, x):
        x = self.output_layer(x)
        x = self.reshape(x)
        out = self.softmax(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


class SumPooling(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
        })
        return config
