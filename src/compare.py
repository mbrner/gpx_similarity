import pathlib
import functools

import click
import geotiler
from geotiler.cache import redis_downloader
import gpxpy
from gpxpy.gpx import GPXTrackPoint
import tensorflow as tf
import redis
from sqlalchemy.orm import sessionmaker
import numpy as np
from scipy.stats import trim_mean

from .create_figs import apply_model, get_nn_model, generate_images_for_segment, bytes2array, decompress_gpx
from .model import get_engine_and_model
from scipy.spatial.distance import cdist, pdist, squareform #(XA, XB, metric='euclidean', *args, **kwargs)







def apply_model_to_file(config, gpx_file, ref_database, weights):
    gpx_file = pathlib.Path(gpx_file)
    embedding_model = get_nn_model(config, weights)
    client = redis.Redis(**config['redis'])
    downloader = redis_downloader(client)
    render_map = functools.partial(geotiler.render_map, downloader=downloader)
    batch_size = config.get('apply').get('batch_size', 16)

    map_options = config['map_options']
    map_options['size'] = (map_options['width'], map_options['height'])
    click.echo('Generating segement images:')
    try:
        gpx = gpxpy.parse(gpx_file.open())
    except gpxpy.gpx.GPXXMLSyntaxException:
        click.echo(f'{gpx_file} is not a valid GPX file!')
        return
    segments = []
    for track_idx, track in enumerate(gpx.tracks):
        for segment_idx, segment in enumerate(track.segments):
            images, entries = [], []
            for img, info in generate_images_for_segment(segment=segment,
                                                         size=map_options['size'],
                                                         zoom=map_options['zoom'],
                                                         max_distance=map_options['smoothing_dist'],
                                                         render_map=render_map,
                                                         show_route=map_options['show_route']):
                entry = {
                    'origin': gpx_file,
                    'track_idx': track_idx,
                    'segment_idx': segment_idx,
                    'image_raw': img}
                entry = {**entry, **info}
                mm = geotiler.Map(extent=(info['p_0_long'], info['p_0_lat'], info['p_1_long'], info['p_1_lat']), zoom=map_options['zoom'])
                entry['point'] = GPXTrackPoint(latitude=mm.center[1], longitude=mm.center[0])
                img_embedding = tf.keras.preprocessing.image.img_to_array(img) / 255.
                images.append(img_embedding)
                entries.append(entry)
                if len(images) == batch_size:
                    images = apply_model(embedding_model, images, batch_size, fill_up=False, func_name='encode')
                    for img, entry in zip(images, entries):
                        entry['image_embedding'] = tf.reshape(img, [np.prod(img.shape),]).numpy().flatten()
                        segments.append(entry)
                    images, entries = [], []
            if len(entries) > 0:
                images = apply_model(embedding_model, images, batch_size, fill_up=True, func_name='encode')
                for img, entry in zip(images, entries):
                    entry['image_embedding'] = tf.reshape(img, [np.prod(img.shape),]).numpy().flatten()
                    segments.append(entry)
    test_images = np.asarray([entry['image_embedding'] for entry in segments])
    print(test_images.shape)
    click.echo('Loading Segments from Database')
    engine, Routes, Segments = get_engine_and_model(ref_database, train=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    images = []
    route_ids = [] 
    for seg in session.query(Segments.image, Segments.origin):
        images.append(bytes2array(seg.image))
        route_ids.append(seg.origin)
    images = np.asarray(images)
    routes_ids = np.asarray(route_ids)
    matrix = cdist(test_images, images, metric=config['apply']['metric'])
    ref_routes = np.unique(routes_ids)
    ref_files = {i: session.query(Routes.path).filter(Routes.id == int(i)).first().path for i in ref_routes}

    
    if config['apply']['aggregation'].lower().strip() == 'mean':
        aggregation_func = np.mean
    elif config['apply']['aggregation'].lower().strip() == 'median':
        aggregation_func = np.median
    elif config['apply']['aggregation'].lower().startswith('trim_mean'):
        _, _, percentage = config['apply']['aggregation'].partition('::')
        aggregation_func = functools.partial(trim_mean, proportiontocut=float(percentage))
    else:
        raise ValueError(f"{config['apply']['aggregation']} is not a valid aggregation function!")
    mean_dist = np.empty(len(ref_routes), dtype=float)
    for i, idx_i in enumerate(ref_routes):
        mask_i = route_ids == idx_i
        dist_ij = matrix[:, mask_i]
        min_dist = np.min(dist_ij, axis=1)
        mean_dist[i] = aggregation_func(min_dist)
    mean_dist[mean_dist < 1E-10] = np.inf
    idx_min = np.argmin(mean_dist)
    path_most_sim = ref_files[ref_routes[idx_min]]
    entry = session.query(Routes).filter(Routes.path == path_most_sim).first()
    