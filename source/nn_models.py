import sys, re, pathlib

import tensorflow as tf
from tensorflow.keras import layers, models 


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
                 *args, **kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        store_attr()
        if len(pooling_layer_opts) != len(conv_layer_opts):
            raise ValueError(f'`conv_layer_opts` [len={len(conv_layer_opts)}] and `pooling_layer_opts`'
                             f' [len={len(pooling_layer_opts)}] of unequal length!')
        self.n_layers_per_side = 0
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
            setattr(self, f'conv_{i}', layers.Conv2D(**opts))
            setattr(self, f'pool_{i}', layers.MaxPooling2D(pool_i))
            self.n_layers_per_side += 1


        for i in range(1, self.n_layers_per_side+1)[::-1]:
            setattr(self, f'upsamp_{i}', layers.UpSampling2D(size=getattr(self, f'pool_{i}').pool_size))
            setattr(self, f'trans_{i}', layers.Conv2DTranspose.from_config(getattr(self, f'conv_{i}').get_config()))

        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        self.conv_final = layers.Conv2D(3, 3, activation='sigmoid', padding=padding)
        self.flatten = layers.Flatten()

    def encode(self, x, flatten=False):
        for i in range(1, self.n_layers_per_side+1):
            x = getattr(self, f'conv_{i}')(x)
            x = getattr(self, f'pool_{i}')(x)
        if flatten:
            return self.flatten(x)
        else:
            return x
                
    def decode(self, x):
        for i in range(1, self.n_layers_per_side+1)[::-1]:
            x = getattr(self, f'upsamp_{i}')(x)
            x = getattr(self, f'trans_{i}')(x)        
        return self.conv_final(x)
        
        
    def call(self, inputs):
        x = self.encode(inputs, flatten=False)
        if self.dropout:
            x = self.dropout(x)
        return self.decode(x)

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