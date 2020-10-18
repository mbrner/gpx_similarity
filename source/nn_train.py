from io import BytesIO
import pathlib
import datetime
from functools import partial

import toml
import click
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import losses
import numpy as np
from PIL import Image

from sqlalchemy.sql.expression import and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy import text
from sqlalchemy_filters import apply_filters

from .model import generator_from_query_rnd_order, get_engine_and_model, postgres_generator
from .nn_models import Autoencoder
from .nn_model_no_subclassing import create_model
from .visualize import show_comparisons
from .create_figs import SaveType, bytes2array


def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)



def create_array_from_img(result):
    if result.save_type == SaveType.PNG.value:
        buf = BytesIO(result.image)
        img = Image.open(buf).convert('RGB')        
        return tf.keras.preprocessing.image.img_to_array(img) / 255.
    elif result.save_type == SaveType.ARRAY.value:
        return bytes2array(result.image)



def create_array_from_img_train(result):
    if result.save_type == SaveType.PNG.value:
        buf = BytesIO(result.image)
        img = Image.open(buf).convert('RGB')        
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
    elif result.save_type == SaveType.ARRAY.value:
        img= bytes2array(result.image)
    randint = np.random.randint(8)
    k = randint % 4
    flip = bool(randint // 4)
    if k > 0:
        img = tf.image.rot90(img, k=k)
    if flip:
        img = tf.image.flip_left_right(img)
    return img



class TestImagesTensorboard(callbacks.Callback):
    def __init__(self, test_images, writer, max_outputs=10):
        super().__init__() 
        self.images = test_images
        self.max_outputs = max_outputs
        self.writer = writer

    def on_epoch_end(self, epoch, logs={}):
        [reconstructed, _, _] = self.model(self.images)
        with self.writer.as_default():
            tf.summary.image(f'Testimages', np.hstack((self.images, reconstructed.numpy())), step=epoch, max_outputs=self.max_outputs)


def train(config, output_dir, checkpoint=None):
    engine, OSMImages = get_engine_and_model(**config['postgres'])
    Session = sessionmaker(bind=engine)
    session = Session()
    
    opts = config['train']

    output_dir = pathlib.Path(output_dir) / datetime.datetime.now().strftime('%Y%m%d_%H%M')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    query = session.query(OSMImages.image, OSMImages.save_type)
    if filters := config['train'].get('filters', None):
        query = apply_filters(query, filters)
    filter_str = str(query.statement.compile(compile_kwargs={"literal_binds": True})).replace('\n', '\n\t')
    opts['filters'] = f'\n\t{filter_str}'

    output_shape = (config['map_options']['height'], config['map_options']['width'], config['map_options']['channels'])
    generator_train, _  = generator_from_query_rnd_order(
                             query=query,
                             train_test_split=opts['train_test_split'],
                             seed=opts['seed'],
                             test=False,
                             callback=create_array_from_img_train)
    train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(output_shape, output_shape))
    train_dataset_batched = train_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 30)

    generator_test, _ = generator_from_query_rnd_order(
                             query=query,
                             train_test_split=opts['train_test_split'],
                             seed=opts['seed'],
                             test=True,
                             callback=create_array_from_img_train)
    test_dataset = tf.data.Dataset.from_generator(lambda: generator_test(), output_types=(tf.float32, tf.float32), output_shapes=(output_shape, output_shape))
    test_dataset_batched = test_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 3)

    with (output_dir / 'config.toml').open('w') as stream:
        toml.dump(config, stream)
    click.echo('\n'.join(['Options for training:'] + [f'-{k}: {v}' for k, v in opts.items()]))

    model_config = {**config['model'], 'width': config['map_options']['width'], 'height': config['map_options']['height']}
    if checkpoint is None:
        model = create_model(**model_config)
        model.compile(optimizer=tf.keras.optimizers.Adam())
        initial_epoch = 0
    else:
        click.echo(f'Loading model from: {checkpoint}!')
        model = tf.keras.models.load_model(checkpoint)
        try:
            initial_epoch = int(checkpoint.partition('-')[-1].split('.')[0])
        except ValueError:
            initial_epoch = 10
            click.echo('Epoch number unknown!')
    enable_memory_growth()
    learning_rate = opts['learning_rate']
    if isinstance(learning_rate, float):
        learning_rates = [10**learning_rate] * opts['epochs']
        lr_callback = callbacks.LearningRateScheduler(lambda _: learning_rate)
    else:
        try:
            learning_rates = np.logspace(*learning_rate, opts['epochs'])
        except ValueError:
            raise ValueError('`learning_rate` has to either a single float or a tuple of two floats [rate_beginning, rate_end]')
    
    def lr_callback_func(epoch):
        learning_rate = learning_rates[epoch-initial_epoch]
        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate
    lr_callback = callbacks.LearningRateScheduler(lr_callback_func)

    checkpoint_path = output_dir / 'logs' / 'models' / 'cp-{epoch:04d}.ckpt'
    model_checkpoint_callback = callbacks.ModelCheckpoint(str(checkpoint_path))
    writer = tf.summary.create_file_writer(logdir=str(output_dir / 'logs' / 'test'))
    tensorboard_callback = callbacks.TensorBoard(log_dir=output_dir / 'logs')
    images = np.array(*test_dataset_batched.take(1).as_numpy_iterator())[0]

    model.fit(train_dataset_batched,
              initial_epoch=initial_epoch,
              epochs=initial_epoch+opts.get('epochs', 10),
              validation_data=test_dataset_batched,
              callbacks=[lr_callback, tensorboard_callback, TestImagesTensorboard(images, writer), model_checkpoint_callback])