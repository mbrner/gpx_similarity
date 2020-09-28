import tensorflow as tf

from tensorflow.keras import layers, models 


class Autoencoder(tf.keras.Model):

    def __init__(self, width=256, height=256, channels=3, *args, **kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        self.conv_1 = layers.Conv2D(32, 8, activation='relu', padding='same', input_shape=(height, width, channels))
        self.pool_1 = layers.MaxPooling2D(4)
        
        self.conv_2 = layers.Conv2D(16, 4, activation='relu', padding='same')
        self.pool_2 = layers.MaxPooling2D(4)
        
        self.conv_3 = layers.Conv2D(8, 2, activation='relu', padding='same')
        self.pool_3 = layers.MaxPooling2D(2)

        self.dropout = tf.keras.layers.Dropout(.2)
        
        self.upsamp_3 = layers.UpSampling2D(size=self.pool_3.pool_size)
        self.trans_3 = layers.Conv2DTranspose.from_config(self.conv_3.get_config())
        
        self.upsamp_2 = layers.UpSampling2D(size=self.pool_2.pool_size)
        self.trans_2 = layers.Conv2DTranspose.from_config(self.conv_2.get_config())
        
        self.upsamp_1 = layers.UpSampling2D(size=self.pool_1.pool_size)
        self.trans_1 = layers.Conv2DTranspose.from_config(self.conv_1.get_config())
        
        self.conv_final = layers.Conv2D(3, 3, activation='sigmoid', padding='same')

    def encode(self, x, flatten=False):
        x = self.conv_1(x)
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = self.pool_2(x)
        
        x = self.conv_3(x)
        x = self.pool_3(x)
        return x
                
    def decode(self, x):
        x = self.upsamp_3(x)
        x = self.trans_3(x)
        
        x = self.upsamp_2(x)
        x = self.trans_2(x)
        
        x = self.upsamp_1(x)
        x = self.trans_1(x)
        
        return self.conv_final(x)
        
    def call(self, inputs):
        x = self.encode(inputs, flatten=False)
        x = self.dropout(x)
        return self.decode(x)

    @classmethod
    def from_weights(cls, index_file):
        model = cls()
        model.load_weights(index_file)
        return model