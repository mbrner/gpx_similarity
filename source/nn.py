from io import BytesIO
import pathlib
import datetime

import toml
import click
import tensorflow as tf
import numpy as np
from PIL import Image

from sqlalchemy.sql.expression import and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy import text
from sqlalchemy_filters import apply_filters

from .model import generator_from_query_rnd_order, get_engine_and_model, postgres_generator
from .nn_models import Autoencoder
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
        return bytes2array(results.image)



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


def train(config, output_dir, weights=None):
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

    model_config = {**config['model'], 'width': config['map_options']['width'], 'height': config['map_options']['height']}
    model = Autoencoder(**model_config)
    if weights is not None:
        model.load_weights(weights)
        opts['initial_weights'] = str(weights)
    else:
        opts['initial_weights'] = None
    click.echo('\n'.join(['Options for training:'] + [f'-{k}: {v}' for k, v in opts.items()]))
    model.save_weights(str(output_dir / 'models' / f'model_epoch_{"0".zfill(3)}.weights'), overwrite=True)
    with (output_dir / 'config.toml').open('w') as stream:
        toml.dump(config, stream)
    optimizer = tf.keras.optimizers.Adam(learning_rate=opts['learning_rate'])
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()
    
    enable_memory_growth()

    for epoch in range(opts['epochs']):
        click.echo(f"Start of epoch {epoch+1}/{opts['epochs']}")
        generator_train, entries_train = generator_from_query_rnd_order(query,
                                                            callback=create_array_from_img_train,
                                                            train_test_split=0.1,
                                                            seed=1337,
                                                            test=False,
                                                            return_entries=True)
        train_dataset = tf.data.Dataset.from_generator(lambda: generator_train, output_types=tf.float32)

        generator_test, entries_test = generator_from_query_rnd_order(query,
                                                            callback=create_array_from_img,
                                                            train_test_split=0.1,
                                                            seed=1337,
                                                            test=True,
                                                            return_entries=True)
        test_dataset = tf.data.Dataset.from_generator(lambda: generator_test, output_types=tf.float32)


        train_dataset_batched = train_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 3)
        test_dataset_batched = test_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 3)
        click.echo(f'Using {entries_train} samples train and {entries_test} samples test')
        n_steps = int(entries_train / opts['batch_size'])
        for step, x_batch_train in enumerate(train_dataset_batched):
            step += 1
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train)
                loss = mse_loss_fn(x_batch_train, reconstructed)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if (step % 10 == 0) or step == n_steps:
                click.echo(f"step  {step}/{n_steps}: mean loss [train]= {loss_metric.result():.4f}")
        loss_metric.reset_states()
        for step, x_batch_test in enumerate(test_dataset_batched):
            reconstructed = model(x_batch_test)
            loss = mse_loss_fn(x_batch_test, reconstructed)
            loss_metric(loss)
            if step == 0:
                show_comparisons(output_dir / f'maps_{str(epoch+1).zfill(3)}_train.png', x_batch_test, reconstructed, n_rows=4)
        model.save_weights(str(output_dir / 'models' / f'model_epoch_{str(epoch).zfill(3)}.weights'), overwrite=True)
        model.save('test_save')
        click.echo(f"mean loss [test] = {loss_metric.result():.4f}")