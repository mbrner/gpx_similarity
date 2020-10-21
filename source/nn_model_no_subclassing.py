import sys, re, pathlib, math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import tensorflow.keras.backend as K


def _transform_shape_conv2d(shape, strides, filters, kernel_size, padding='same', data_format='channels_last'):
    if data_format == 'channels_last':
        w, h, _ = shape
    else:
        _, w, h = shape
    if padding == 'same':
        new_w = w
        new_h = h
    elif padding == 'valid':
        new_w = w - kernel_size[0] + 1
        new_h = h - kernel_size[1] + 1
    new_w = int(math.ceil(new_w / strides[0]))
    new_h = int(math.ceil(new_h / strides[1]))
    if data_format == 'channels_last':
        return new_w, new_h, filters
    else:
        return filters, new_w, new_h
    

def _transform_shape_maxpool2d(shape, strides, pool_size, padding='same', data_format='channels_last'):  
    if data_format == 'channels_last':
        w, h, c = shape
    else:
        c, w, h = shape
    if padding == 'same':
        new_w = w
        new_h = h
    elif padding == 'valid':
        new_w = w - pool_size[0] + 1
        new_h = h - pool_size[1] + 1
    new_w = int(math.ceil(new_w / strides[0]))
    new_h = int(math.ceil(new_h / strides[1]))
    if data_format == 'channels_last':
        return new_w, new_h, c
    else:
        return c, new_w, new_h


def transform_shape_conv2d(input_shape, l):
    return _transform_shape_conv2d(input_shape, l.strides, l.filters, l.kernel_size, l.padding, l.data_format)


def transform_shape_maxpool2d(input_shape, l):
    return _transform_shape_maxpool2d(input_shape, l.strides, l.pool_size, l.padding, l.data_format)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * epsilon
       

def create_model(width=256,
                 height=256,
                 channels=3,
                 activation='relu',
                 padding='same',
                 dropout_rate=0.2,
                 conv_layer_opts=[(32, 8), (16, 4), (8, 2)],
                 pooling_layer_opts=[4, 4, 2],
                 mse_loss_factor=None,):
    input_shape = (height, width, channels)
    encoder_input = tf.keras.Input(shape=input_shape, name='encoder_input')
    conv_pool_layers = []
    for i, (conv_i, pool_i) in enumerate(zip(conv_layer_opts, pooling_layer_opts)):
        opts = {'filters': conv_i[0],
                'kernel_size': conv_i[1],
                'activation': activation,
                'padding': padding}
        l_conv = layers.Conv2D(name=f'conv_{i}', **opts)
        l_pool = layers.MaxPooling2D(pool_i, name=f'pool_{i}')
        if i == 0:
            x = l_conv(encoder_input)
        else:
            x = l_conv(x)
        x = l_pool(x)
        input_shape = transform_shape_conv2d(input_shape, l_conv)
        input_shape = transform_shape_maxpool2d(input_shape, l_pool)
        conv_pool_layers.append((l_conv, l_pool))
    latent_shape = input_shape
    latent_dim = np.prod(input_shape)
    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim)(x)
    log_var = layers.Dense(latent_dim)(x)
    x = Sampling()([mu, log_var])
    encoder_output = layers.Reshape(latent_shape)(x)

    decoder_input = tf.keras.Input(shape=latent_shape, name='decoder_input')
    decoder_input = encoder_output
    if dropout_rate > 0:
        dropout = tf.keras.layers.Dropout(dropout_rate)
        x = dropout(decoder_input)    
    else:
        x = decoder_input
    for i, (l_conv, l_pool) in enumerate(conv_pool_layers):
        l_up = layers.UpSampling2D(size=l_pool.pool_size)
        cfg = l_conv.get_config()
        cfg['name'] = f'conv_trans_{i}'
        l_conv_trans = layers.Conv2DTranspose.from_config(cfg)
        x = l_up(x)
        x = l_conv_trans(x)
    conv_final = layers.Conv2D(3, 3, activation='sigmoid', padding=padding)
    decoder_output = conv_final(x)
    full_vae = models.Model(encoder_input, [decoder_output, mu, log_var])

    reconstruction_loss = tf.math.reduce_mean((encoder_input-decoder_output)**2, axis=[1,2,3]) * mse_loss_factor
    kl_loss = 1 + log_var - tf.math.square(mu) - tf.math.exp(log_var)
    kl_loss = tf.math.reduce_mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    full_vae.add_loss(K.mean(reconstruction_loss))
    full_vae.add_loss(K.mean(kl_loss))

    return full_vae
