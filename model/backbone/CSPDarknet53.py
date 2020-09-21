from typing import Tuple

import tensorflow as tf

from model.layer import CSPStage, MyConv2D


class CSPDarknet53(tf.keras.Model):
    def __init__(self, name="CSPDarknet53", **kwargs):
        super(CSPDarknet53, self).__init__(name=name, **kwargs)
        self.conv = MyConv2D(filters=32, kernel_size=3, activation="mish", apply_dropblock=True)

        self.stages = [
            CSPStage(filters=[32, 64], num_blocks=1),
            CSPStage(filters=[64, 128], num_blocks=2),
            CSPStage(filters=[128, 256], num_blocks=8),
            CSPStage(filters=[256, 512], num_blocks=8),
            CSPStage(filters=[512, 1024], num_blocks=4)
        ]

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.conv(inputs, training=training)
        # CSP stage 1
        x = self.stages[0](x, training=training)

        # CSP stage 2
        x = self.stages[1](x, training=training)

        # CSP stage 3
        x = self.stages[2](x, training=training)
        output_large = x  # lg scale output

        # CSP stage 4
        x = self.stages[3](x, training=training)
        output_medium = x  # md scale output

        # CSP stage 5
        x = self.stages[4](x, training=training)
        output_small = x  # sm scale output

        return output_small, output_medium, output_large
