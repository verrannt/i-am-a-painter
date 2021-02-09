import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from .generic import downsample, upsample, residual_block, ReflectionPadding2D

class Generators():

    def __init__(self):
        pass

    def unet(output_channels=3):
        inputs = layers.Input(shape=[256,256,3])

        # bs = batch size
        down_stack = [
            downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
            downsample(128, 4), # (bs, 64, 64, 128)
            downsample(256, 4), # (bs, 32, 32, 256)
            downsample(512, 4), # (bs, 16, 16, 512)
            downsample(512, 4), # (bs, 8, 8, 512)
            downsample(512, 4), # (bs, 4, 4, 512)
            downsample(512, 4), # (bs, 2, 2, 512)
            downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            upsample(512, 4), # (bs, 16, 16, 1024)
            upsample(256, 4), # (bs, 32, 32, 512)
            upsample(128, 4), # (bs, 64, 64, 256)
            upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(output_channels, 4,
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    activation='tanh') # (bs, 256, 256, 3)
    
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        x = last(x)

        return keras.Model(inputs=inputs, outputs=x)

    def resnet(
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsample_blocks=2,
        name=None,
    ):
        kernel_init = tf.random_normal_initializer(0., 0.02)
        gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        inputs = layers.Input(shape=[256,256,3])
        
        # First block
        x = ReflectionPadding2D(padding=(3, 3))(inputs)
        x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = downsample(filters, 3)(x)

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = residual_block(x, activation=layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = upsample(filters, 3)(x)

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(3, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        model = keras.models.Model(inputs, x, name=name)
        return model
