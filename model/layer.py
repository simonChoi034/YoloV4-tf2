from typing import Union, List

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, Concatenate, MaxPool2D, UpSampling2D, \
    Activation, ReLU
from tensorflow_addons.layers.normalizations import GroupNormalization


class DropBlock(Layer):
    def __init__(self, keep_prob, block_size, name="dropblock", **kwargs):
        super(DropBlock, self).__init__(name=name, **kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        bottom = right = (self.block_size - 1) // 2
        top = left = (self.block_size - 1) - bottom
        self.padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
        self.set_keep_prob()
        super(DropBlock, self).build(input_shape)

    def call(self, inputs, training=None, scale=True, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, tf.float32), tf.cast(self.h, tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = DropBlock._bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

    @staticmethod
    def _bernoulli(shape, mean):
        return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class Mish(Layer):
    def __init__(self, name="mish", **kwargs):
        super(Mish, self).__init__(name=name, **kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return inputs * tf.math.tanh(tf.math.softplus(inputs))


def create_activation_layer(name: str) -> Union[Mish, LeakyReLU, ReLU, Activation]:
    if name == "mish":
        return Mish()
    elif name == "leaky_relu":
        return LeakyReLU(alpha=0.1)
    elif name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Activation(tf.nn.sigmoid)
    else:
        return Activation("linear", dtype=tf.float32, name="Output-float32-casting")


class MyConv2D(Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: Union[List, int],
            strides: int = 1,
            dilation_rate: float = 1,
            padding: str = "same",
            activation: Union[None, str] = "leaky_relu",
            apply_batchnorm: bool = True,
            apply_dropblock: bool = False,
            keep_prob: float = 0.8,
            dropblock_size: int = 3,
            name: str = "conv2d",
            **kwargs):
        super(MyConv2D, self).__init__(name=name, **kwargs)
        self.conv2d = Conv2D(
            filters,
            kernel_size,
            strides,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=tf.initializers.GlorotNormal(),
            use_bias=False
        )
        self.activation = create_activation_layer(activation)
        self.apply_activation = activation is not None
        self.apply_batchnorm = apply_batchnorm
        self.apply_dropblock = apply_dropblock
        self.batch_norm = GroupNormalization(groups=32)
        self.drop_block = DropBlock(keep_prob=keep_prob, block_size=dropblock_size)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = self.conv2d(inputs)
        if self.apply_batchnorm:
            x = self.batch_norm(x, training=training)

        if self.apply_activation:
            x = self.activation(x)

        if self.apply_dropblock:
            x = self.drop_block(x, training=training)

        return x


class CSPBlock(Layer):
    def __init__(self, filters: Union[List, int], name: str = "CSPBlock", **kwargs):
        super(CSPBlock, self).__init__(name=name, **kwargs)
        self.filters = [filters, filters] if isinstance(filters, int) else filters
        self.convs = Sequential([
            MyConv2D(filters=self.filters[0], kernel_size=1, activation="mish", apply_dropblock=True, name="csp_conv1"),
            MyConv2D(filters=self.filters[1], kernel_size=3, activation="mish", apply_dropblock=True, name="csp_conv2")
        ])

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        x = self.convs(inputs, training=training)
        # residual shortcut
        x += inputs

        return x


class CSPStage(Layer):
    def __init__(self, filters: Union[List, int], num_blocks: int, name="CSPStage", **kwargs):
        super(CSPStage, self).__init__(name=name, **kwargs)
        self.filters = [filters, filters] if isinstance(filters, int) else filters

        # down_sampling conv
        self.down_sampling = MyConv2D(self.filters[1], kernel_size=3, strides=2, activation="mish",
                                      apply_dropblock=True)

        # csp split conv
        self.split_conv_1 = MyConv2D(filters=self.filters[0], kernel_size=1, activation="mish", apply_dropblock=True)
        self.split_conv_2 = MyConv2D(filters=self.filters[0], kernel_size=1, activation="mish", apply_dropblock=True)

        # residual conv block
        self.conv_blocks = Sequential(
            [CSPBlock(filters=self.filters[0], name="csp_block_{}".format(i + 1)) for i in range(num_blocks)]
        )
        self.conv1x1 = MyConv2D(filters=self.filters[0], kernel_size=1, activation="mish", apply_dropblock=True)

        # stage output conv
        self.concat_conv = MyConv2D(filters=self.filters[1], kernel_size=1, activation="mish", apply_dropblock=True)

        self.concat = Concatenate()

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        # down_sampling
        x = self.down_sampling(inputs, training=training)

        # split conv for csp
        x1 = self.split_conv_1(x, training=training)
        x2 = self.split_conv_2(x, training=training)

        # csp blocks
        x2 = self.conv_blocks(x2, training=training)
        x2 = self.conv1x1(x2, training=training)

        # cross stage connection
        x = self.concat([x1, x2])
        x = self.concat_conv(x, training=training)

        return x


class SpatialPyramidPooling(Layer):
    def __init__(self, pool_sizes: List = [5, 9, 13], name="SPP", **kwargs):
        super(SpatialPyramidPooling, self).__init__(name=name, **kwargs)
        self.poolings = [MaxPool2D(pool_size=pool_size, strides=1, padding="same") for pool_size in pool_sizes]
        self.concat = Concatenate()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        features = [max_pooling(inputs) for max_pooling in self.poolings]
        features = self.concat([inputs] + features)

        return features


class UpSampling(Layer):
    def __init__(self, filters: Union[List, int], size: int = 2, apply_dropblock: bool = False, name="up_sampling",
                 **kwargs):
        super(UpSampling, self).__init__(name=name, **kwargs)
        self.up_sampling = Sequential([
            UpSampling2D(size=size),
            MyConv2D(filters=filters, kernel_size=1, apply_dropblock=apply_dropblock)
        ])

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        return self.up_sampling(inputs, training=training)


class DownSampling(Layer):
    def __init__(self, filters: Union[List, int], size: int = 2, apply_dropblock: bool = False, name="down_sampling",
                 **kwargs):
        super(DownSampling, self).__init__(name=name, **kwargs)
        self.down_sampling = MyConv2D(filters=filters, kernel_size=3, strides=size, apply_dropblock=apply_dropblock)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        return self.down_sampling(inputs, training=training)


class SpatialAttention(Layer):
    def __init__(self, name='spatial-attention', **kwargs):
        super(SpatialAttention, self).__init__(name=name, **kwargs)
        self.spatial_conv = MyConv2D(filters=1, kernel_size=7, activation="sigmoid", apply_dropblock=False,
                                     apply_batchnorm=False)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs):
        # spatial attention
        y = self.spatial_conv(inputs, training=training)
        y = tf.multiply(inputs, y)

        return y
