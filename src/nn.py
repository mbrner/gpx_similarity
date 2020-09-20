from io import BytesIO

import click
import tensorflow as tf
from PIL import Image
from sqlalchemy.sql.expression import and_
from sqlalchemy.orm import sessionmaker

from .model import generator_from_query, get_engine_and_model
from .nn_models import Autoencoder
from .visualize import show_comparisons


def create_array_from_img(result):
    buf = BytesIO(result.image)
    img = Image.open(buf).convert('RGB')        
    return tf.keras.preprocessing.image.img_to_array(img) / 255.


def train(config):
    engine, OSMImages = get_engine_and_model(**config['postgres'])
    Session = sessionmaker(bind=engine)
    session = Session()
    
    opts = config['train']

    query = session.query(OSMImages).filter(and_(OSMImages.route_type == 'bike', OSMImages.dataset == 'komoot'))
    click.echo('\n'.join(['Options for training:'] + [f'-{k}: {v}' for k, v in opts.items()]))

    model = Autoencoder(width=config['map_options']['width'], height=config['map_options']['height'])
    #model.save_weights(f'geolife_bike_train/trained_models/model_epoch_{"0".zfill(3)}.h5', overwrite=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=opts['learning_rate'])
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()


    # Iterate over epochs.
    for epoch in range(opts['epochs']):
        click.echo(f"Start of epoch {epoch+1}/{opts['epochs']}")
        generator_train, entries_train = generator_from_query(query,
                                                            callback=create_array_from_img,
                                                            train_test_split=0.1,
                                                            seed=1337,
                                                            test=False,
                                                            return_entries=True)
        train_dataset = tf.data.Dataset.from_generator(lambda: generator_train, output_types=tf.float32)

        generator_test, entries_test = generator_from_query(query,
                                                            callback=create_array_from_img,
                                                            train_test_split=0.1,
                                                            seed=1337,
                                                            test=True,
                                                            return_entries=True)
        test_dataset = tf.data.Dataset.from_generator(lambda: generator_test, output_types=tf.float32)


        train_dataset_batched = train_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 3)
        test_dataset_batched = test_dataset.batch(opts['batch_size'], drop_remainder=True).prefetch(opts['batch_size'] * 3)
        click.echo(f'Using {entries_train} samples train and {entries_test} samples test')
        for step, x_batch_train in enumerate(train_dataset_batched):
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train)
                loss = mse_loss_fn(x_batch_train, reconstructed)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 10 == 0:
                click.echo(f"step  {step}/{int(entries_train)/opts['batch_size']}: mean loss [train]= {loss_metric.result():.4f}")
        print('Running test set!')
        loss_metric.reset_states()
        for step, x_batch_test in enumerate(test_dataset_batched):
            reconstructed = model(x_batch_test)
            loss = mse_loss_fn(x_batch_test, reconstructed)
            loss_metric(loss)
            if step == 0:
                show_comparisons(f'geolife_bike_train/model_{str(epoch+1).zfill(3)}_train.png', x_batch_test, reconstructed)
        #model.save_weights(f'geolife_bike_train/trained_models/model_epoch_{str(epoch+1).zfill(3)}.h5', overwrite=True)
        click.echo(f"mean loss [test] = {loss_metric.result():.4f}")