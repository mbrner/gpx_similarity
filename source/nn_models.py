import sys, re, pathlib, math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models 


def _transform_shape_conv2d(shape, strides, filters, kernel_size, padding='same', data_format='channels_last'):
    if data_format == 'channels_last':
        w, h, c = shape
    else:
        c, w, h = shape
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


def _store_attr(self, args, store_dict=None, assign=True, ):
    for n,v in args.items():
        if assign:
            setattr(self, n, v)
        store_dict[n] = v


def store_attr(names=None, self=None, assign=False, but=None, store_dict='__stored_args__', **attrs):
    "Store params named in comma-separated `names` from calling context into attrs in `self`"
    fr = sys._getframe(1)
    args = fr.f_code.co_varnames[:fr.f_code.co_argcount]
    if self: args = ('self', *args)
    else: self = fr.f_locals[args[0]]
    if isinstance(store_dict, dict):
        pass
    elif store_dict is None:
        store_dict = {}
    elif isinstance(store_dict, str):
        if not hasattr(self, store_dict):
            setattr(self, store_dict, {})
            store_dict = getattr(self, store_dict)
    else:
        raise ValueError('"store_dict" has to be str or dict')
    if attrs:
        return _store_attr(self, attrs, store_dict, assign)
    ns = re.split(', *', names) if names else args[1:]
    if but:
        but = set(but)
    else:
        but = set()
    _store_attr(self, {n:fr.f_locals[n] for n in ns if n not in but}, store_dict, assign)
    return store_dict


class VAEBlock(tf.keras.layers.Layer):
    def __init__(self, kl_loss_factor, latent_shape, file_writer=None, total_batches=None):
        super(VAEBlock, self).__init__()
        self.latent_dim = np.prod(latent_shape)
        self.latent_shape = latent_shape
        self.flatten = layers.Flatten()
        self.reshape = layers.Reshape(self.latent_shape)
        self.mu = layers.Dense(self.latent_dim, name='mu')
        self.log_var = layers.Dense(self.latent_dim, name='log_var')
        self.kl_loss_factor = kl_loss_factor
        self.file_writer = file_writer
        self.total_batches = total_batches

    def call(self, inputs):
        x = self.flatten(inputs)
        mu = self.mu(x)
        log_var = self.log_var(x)
        loss = -0.5 * tf.reduce_sum(1 + log_var - mu**2 - tf.exp(log_var))
        if self.file_writer is None and self.total_batches is not None:
            with self.file_writer.as_default():
                tf.summary.scalar("kl_loss", loss, step=self.total_batches)
        self.add_loss(loss)
        return self.reparameterize(mu, log_var)

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=mu.shape)
        x = eps * tf.exp(log_var * .5) + mu
        return self.reshape(x)

class Autoencoder(tf.keras.Model):

    def __init__(self,
                 width=256,
                 height=256,
                 channels=3,
                 activation='relu',
                 padding='same',
                 dropout_rate=0.2,
                 conv_layer_opts=[(32, 8), (16, 4), (8, 2)],
                 pooling_layer_opts=[4, 4, 2],
                 kl_loss_factor=None,
                 file_writer=None,
                 *args, **kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        store_attr(but=['file_writer'])
        self.epoch_num = tf.Variable(0, trainable=False)
        self.step_num = tf.Variable(0, trainable=False)
        self.total_batches = tf.Variable(0, trainable=False)

        if len(pooling_layer_opts) != len(conv_layer_opts):
            raise ValueError(f'`conv_layer_opts` [len={len(conv_layer_opts)}] and `pooling_layer_opts`'
                             f' [len={len(pooling_layer_opts)}] of unequal length!')
        self.n_layers_per_side = 0
        input_shape = (height, width, channels)
        self.initial_shape = input_shape
        for i, (conv_i, pool_i) in enumerate(zip(conv_layer_opts, pooling_layer_opts)):
            i += 1
            opts = {
                'filters': conv_i[0],
                'kernel_size': conv_i[1],
                'activation': activation,
                'padding': padding,
            }
            if i == 0:
                opts['input_shape'] = (height, width, channels)
            l_conv = layers.Conv2D(**opts)
            l_pool = layers.MaxPooling2D(pool_i)
            setattr(self, f'conv_{i}', l_conv)
            setattr(self, f'pool_{i}', l_pool)
            input_shape = transform_shape_conv2d(input_shape, l_conv)
            input_shape = transform_shape_maxpool2d(input_shape, l_pool)
            self.n_layers_per_side += 1
        self.latent_shape = input_shape
        self.latent_dim = np.prod(input_shape)
        if kl_loss_factor is not None:
            self.vae_block = VAEBlock(kl_loss_factor, self.latent_shape, file_writer=file_writer, total_batches=self.total_batches)
        else:
            self.vae_block = None
        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
        for i in range(1, self.n_layers_per_side+1)[::-1]:
            setattr(self, f'upsamp_{i}', layers.UpSampling2D(size=getattr(self, f'pool_{i}').pool_size))
            setattr(self, f'trans_{i}', layers.Conv2DTranspose.from_config(getattr(self, f'conv_{i}').get_config()))
        self.conv_final = layers.Conv2D(3, 3, activation='sigmoid', padding=padding)
        self.file_writer = file_writer


    def encode(self, x):
        for i in range(1, self.n_layers_per_side+1):
            conv_l = getattr(self, f'conv_{i}')
            x = conv_l(x)
            x = getattr(self, f'pool_{i}')(x)
        if self.vae_block:
            x = self.vae_block(x)
        return x
     
    def decode(self, x):
        for i in range(1, self.n_layers_per_side+1)[::-1]:
            x = getattr(self, f'upsamp_{i}')(x)
            x = getattr(self, f'trans_{i}')(x)        
        return self.conv_final(x)
        
    def call(self, inputs):
        x = self.encode(inputs)
        if self.dropout:
            x = self.dropout(x)
        result = self.decode(x)
        if self.file_writer:
            with self.file_writer.as_default():
                #tf.summary.image("Training data", inputs, step=0, max_outputs=25)
                #tf.summary.image("Decoded data", result, step=0, max_outputs=25)
                #tf.summary.scalar("")
                pass
        return result

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=mu.shape)
        x = eps * tf.exp(log_var * .5) + mu
        return self.reshape(x)

    def apply(self, x):
        return self.reparameterize(*self.encode(x))

    @staticmethod
    def kl_loss(mu, log_var, raxis=1):
        return -0.5 * tf.reduce_sum(1 + log_var - mu**2 - tf.exp(log_var), axis=raxis)

    @classmethod
    def from_config(cls, config, weights=None):
        model = cls(**config)
        if weights is not None:
            weights = pathlib.Path(weights)
            if weights.suffixes[-1] == '.index':
                weights = weights.parent / weights.stem
            model.load_weights(str(weights))
        return model

    def get_config(self):
        config = {}
        for k, v in self.__stored_args__.items():
            type_v = type(v)
            if issubclass(type_v, dict):
                v = dict(v)
            elif issubclass(type_v, list):
                v = list(v)
            elif issubclass(type_v, tuple):
                v = tuple(v)
            config[k] = v
        return config