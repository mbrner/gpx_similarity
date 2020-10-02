import csv
import pathlib
import datetime
import functools
import io
import gzip
import enum

import click
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

import gpxpy
from gpxpy.geo import simplify_polyline
from gpxpy.gpx import GPXTrackPoint
import geotiler
from geotiler.cache import redis_downloader
import redis
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
from PIL import Image


from .model import get_engine_and_model
from .nn_models import Autoencoder


class SaveType(enum.Enum):
    ARRAY = enum.auto()
    PNG = enum.auto()
    NN = enum.auto()


def array2bytes(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return out.read()


def bytes2array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def compress_gpx(path):
    return gzip.compress(path.open('rb').read())


def decompress_gpx(blob):
    return gzip.decompress(blob).decode('utf-8')


def points2array(points):
    lat, long = np.empty(len(points), dtype=float), np.empty(len(points), dtype=float)
    for i, p in enumerate(points):
        lat[i] = p.latitude
        long[i] = p.longitude
    return lat, long


def df_from_points(points):
    df = pd.DataFrame(columns=['latitude', 'longitude', 'elevation', 'time', 'distance', 'distance_total'])
    prev_point = None
    for i,p in enumerate(points):
        dist = p.distance_2d(prev_point) if prev_point is not None else None
        df.loc[i] = [p.latitude,
                     p.longitude,
                     p.elevation,
                     p.time,
                     dist if prev_point is not None else None,
                     df.loc[i-1, 'distance_total'] + dist if prev_point is not None else 0.]
        prev_point = p
    return df


def remove_existing_entries_images(session, model, origin, show_route, zoom, size):
    width, height = size
    delete_q = model.__table__.delete().where(model.path==origin and
                                              model.show_route==show_route and
                                              model.zoom==zoom and
                                              model.width == width and
                                              model.height==height)
    session.execute(delete_q)
    session.commit()

def create_images_train(dbsession,
                         model,
                         gpx_file,
                         show_route=True,
                         route_type='unknown',
                         dataset='unknown',
                         route_id=None,
                         zoom=16,
                         size=(256, 256),
                         max_distance=5,
                         render_map=geotiler.render_map,
                         dpi=100,
                         save_type='array',
                         n_images_per_submit=32):
    if isinstance(save_type, str):
        save_type = SaveType[save_type.upper()]
    elif isinstance(save_type, int):
        save_type = SaveType(save_type)
    elif isinstance(save_type, SaveType):
        pass
    else:
        raise ValueError('`save_type` has to be of type int, str or SaveType')
    remove_existing_entries_images(session=dbsession,
                                   model=model,
                                   origin=str(gpx_file),
                                   show_route=show_route,
                                   zoom=zoom,
                                   size=size)
    try:
        gpx = gpxpy.parse(gpx_file.open())
    except gpxpy.gpx.GPXXMLSyntaxException:
        click.echo(f'{gpx_file} is not a valid GPX file!')
        return
    for track_idx, track in enumerate(gpx.tracks):
        for segment_idx, segment in enumerate(track.segments):
            new_entries = []
            for img, img_info in generate_images_for_segment(segment=segment,
                                                             size=size,
                                                             zoom=zoom,
                                                             max_distance=max_distance,
                                                             render_map=render_map,
                                                             show_route=show_route):
                if save_type == SaveType.ARRAY:
                    img = tf.keras.preprocessing.image.img_to_array(img) / 255.
                    img_bytes = array2bytes(img)
                elif save_type == SaveType.PNG:
                    buf = io.BytesIO()
                    img.save(buf, format='png')
                    buf.seek(0)
                    img_bytes = buf.read()
                else:
                    raise ValueError(f'Invalid `save_type`: {save_type}')
                entry = {
                    'origin': str(gpx_file),
                    'track_idx': track_idx,
                    'segment_idx': segment_idx,
                    'image': img_bytes,
                    'route_type': route_type,
                    'dataset': dataset,
                    'save_type': save_type.value}
                entry = {**entry, **img_info}
                new_entries.append(model(**entry))
                if len(new_entries) >= n_images_per_submit:
                    dbsession.add_all(new_entries)
                    new_entries = []
                    dbsession.commit()


            if len(new_entries) > 0:
                dbsession.add_all(new_entries)
                new_entries = []
                dbsession.commit()


def get_nn_model(config, weights):
    from .nn_model import Autoencoder
    model_config = config['model']
    model_config['width'] = config['map_options']['width']
    model_config['height'] = config['map_options']['height']
    model = Autoencoder.from_config(model_config, weights)
    return model

    
def apply_model(model, batch, batch_size, func_name='call', fill_up=True):
    batch_len = len(batch)
    if batch_len < batch_size and fill_up:
        batch += [batch[0]] * (batch_size - batch_len)
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    result = getattr(model, func_name)(batch)
    if batch_len < batch_size and fill_up:
        return result[:batch_len]
    else:
        return result


def create_images_reference(dbsession,
                            db_model,
                            config,
                            embedding_model,
                            gpx_file,
                            route_id,
                            save_type='nn',
                            batch_size=16,
                            render_map=geotiler.render_map):
    if isinstance(save_type, str):
        save_type = SaveType[save_type.upper()]
    elif isinstance(save_type, int):
        save_type = SaveType(save_type)
    elif isinstance(save_type, SaveType):
        pass
    else:
        raise ValueError('`save_type` has to be of type int, str or SaveType')
    map_options = config['map_optionis']
    remove_existing_entries_images(session=dbsession,
                                   model=db_model,
                                   origin=route_id,
                                   show_route=map_options['show_route'],
                                   zoom=map_options['zoom'],
                                   size=map_options['size'],)
    try:
        gpx = gpxpy.parse(gpx_file.open())
    except gpxpy.gpx.GPXXMLSyntaxException:
        click.echo(f'{gpx_file} is not a valid GPX file!')
        return
    map_options['size'] = (map_options['width'], map_options['height'])
    for track_idx, track in enumerate(gpx.tracks):
        for segment_idx, segment in enumerate(track.segments):
            images, entries = [], []
            for img, img_info in generate_images_for_segment(segment=segment,
                                                             size=map_options['size'],
                                                             zoom=map_options['zoom'],
                                                             max_distance=map_options['max_distance'],
                                                             render_map=render_map,
                                                             show_route=map_options['show_route']):
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.
                entry = {
                    'origin': str(gpx_file),
                    'track_idx': track_idx,
                    'segment_idx': segment_idx,
                    'image': None,
                    'save_type': save_type.value}
                entry = {**entry, **img_info}
                entries.append(entry)
                images.append(img)
                if len(entries) == batch_size:
                    images = apply_model(embedding_model, images, batch_size, fill_up=False, func_name='encode')
                    for i, entry in entries:
                        entry['image'] = array2bytes(tf.reshape(images[i], [np.prod(images[i].shape),]).numpy())
                    dbsession.add_all(entries)
                    dbsession.commit()
                    images, entries = [], []


            if len(entries) > 0:
                images = apply_model(embedding_model, images, batch_size, fill_up=False, func_name='encode')
                for i, entry in entries:
                    entry['image'] = array2bytes(tf.reshape(images[i], [np.prod(images[i].shape),]).numpy())
                dbsession.add_all(entries)
                dbsession.commit()




def generate_images_for_segment(segment,
                                zoom=16,
                                size=(256, 256),
                                max_distance=5,
                                render_map=geotiler.render_map,
                                show_route=False,
                                dpi=100):
    width, height = size
    raw_points = segment.points
    if max_distance is not None:
        points = simplify_polyline(raw_points, max_distance=max_distance)
    else:
        points = raw_points
    df = df_from_points(points)
    f_lat, f_long = interp1d(df.distance_total, df.latitude), interp1d(df.distance_total, df.longitude)
    mm_zero = geotiler.Map(center=(points[0].longitude, points[0].latitude), zoom=zoom, size=size)
    p_0_long, p_0_lat, p_1_long, p_1_lat = mm_zero.extent
    distance = GPXTrackPoint(latitude=p_0_lat, longitude=p_1_long).distance_2d(GPXTrackPoint(latitude=p_1_lat, longitude=p_0_long))
    steps = np.arange(df.distance_total.values[0], df.distance_total.values[-1], distance / (2.*1.41421356237))
    if show_route:
        raw_points_lat, raw_points_long = points2array(raw_points)
    for i, d in enumerate(steps):
        p_lat = f_lat(d)
        p_long = f_long(d)
        mm = geotiler.Map(center=(p_long, p_lat), zoom=zoom, size=size)
        p_0_long, p_0_lat, p_1_long, p_1_lat = mm.extent
        img = render_map(mm)
        if show_route:
            fig = plt.figure(figsize=(width/dpi, height/dpi))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img)
            idx_lat = np.logical_and(p_0_lat <= raw_points_lat, p_1_lat >= raw_points_lat)
            idx_long = np.logical_and(p_0_long <= raw_points_long, p_1_long >= raw_points_long)
            idx = np.logical_and(idx_lat, idx_long)
            selected_lat, selected_long = raw_points_lat[idx], raw_points_long[idx]
            track = np.array([mm.rev_geocode(p) for p in zip(selected_long, selected_lat)])
            if len(track.shape) > 1:
                ax.plot(track[:, 0], track[:, 1], 'r-', alpha=1.0, lw=10, ms=10)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            plt.close(fig)
        else:
            img = img.convert('RGB')
        image_info = dict(
            zoom=zoom,
            idx=i,
            p_0_lat=p_0_lat,
            p_0_long=p_0_long,
            p_1_lat=p_1_lat,
            p_1_long=p_1_long,
            width=width,
            height=height,
            show_route=show_route)
        yield img, image_info
    

def add_train_files(config, in_files, dataset_name, default_route_type, extract_route_type, expand_paths, skip_existing=False):
    engine, OSMImages = get_engine_and_model(**config['postgres'])
    Session = sessionmaker(bind=engine)
    session = Session()

    client = redis.Redis(**config['redis'])
    downloader = redis_downloader(client)
    render_map = functools.partial(geotiler.render_map, downloader=downloader)

    in_files_prepared = []
    for p in in_files:
        p = pathlib.Path(p)
        if expand_paths:
            p = p.absolute()
        if extract_route_type:
            try:
                route_type = p.name.split('_')[-1].split('.')[0]
            except:
                route_type = default_route_type
            else:
                if route_type == '':
                    route_type = default_route_type
        else:
            route_type = default_route_type
        in_files_prepared.append((p, route_type))

    def show_item(item):
        if item is not None:
            return '{} [type: {}]'.format(str(item[0]), item[1])
        else:
            return ''

    opts = config['map_options']
    with click.progressbar(in_files_prepared, item_show_func=show_item, show_pos=True) as bar:
        for (path, route_type) in bar:
            if skip_existing:
                count = session.query(OSMImages.id).filter(
                    OSMImages.origin == str(path) and
                    OSMImages.width == opts['width'] and
                    OSMImages.height == opts['height'] and
                    OSMImages.zoom == opts['zoom'] and
                    OSMImages.show_route == opts['show_route']).count()
                if count > 0:
                    continue
            create_images_train(session,
                                 OSMImages,
                                 path,
                                 route_type=route_type,
                                 render_map=render_map,
                                 show_route=bool(opts['show_route']),
                                 zoom=opts['zoom'],
                                 size=(opts['width'], opts['width']),
                                 max_distance=opts['smoothing_dist'],
                                 save_type=config['train']['save_type'],
                                 dataset=dataset_name)


def add_reference_files(config, weights, reference_database, in_files, dataset_name, default_route_type, extract_route_type, expand_paths, skip_existing=False):
    engine, Routes, OSMImages = get_engine_and_model(reference_database, train=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    client = redis.Redis(**config['redis'])
    downloader = redis_downloader(client)
    render_map = functools.partial(geotiler.render_map, downloader=downloader)
 
    embedding_model = get_nn_model(config, weights)

    in_files_prepared = []
    def show_item_gpx(item):
        if item is not None:
            return f'{item}'
        else:
            return ''

    opts = config['map_options']
    click.echo('Adding GPX Files to Database:')
    with click.progressbar(in_files, item_show_func=show_item_gpx, show_pos=True) as bar:
        for p in bar:
            p = pathlib.Path(p)
            if expand_paths:
                p = p.absolute()
            if extract_route_type:
                try:
                    route_type = p.name.split('_')[-1].split('.')[0]
                except:
                    route_type = default_route_type
                else:
                    if route_type == '':
                        route_type = default_route_type
            else:
                route_type = default_route_type
            if skip_existing:
                if session.query(Routes).count() > 0:
                    continue
            else:
                delete_q = Routes.__table__.delete().where(Routes.path==str(path))
                session.execute(delete_q)
                session.commit()
            route_entry = Routes(
                path=str(p),
                dataset=dataset_name,
                route_type=route_type,
                gpx_file=compress_gpx(p))
            session.add(route_entry)
            session.commit()
            in_files_prepared.append((p, route_entry.id))        

    def show_item_image(item):
        if item is not None:
            return '{} [type: {}]'.format(str(item[0]), item[1])
        else:
            return ''

    opts = config['map_options']
    click.echo('Generating segement images:')
    with click.progressbar(in_files_prepared, item_show_func=show_item_image, show_pos=True) as bar:
        for (path, route_entry_id) in bar:
            create_images_reference(dbsession=session,
                                    db_model=OSMImages,
                                    confg=config,
                                    embedding_model=embedding_model,
                                    gpx_file=path,
                                    route_id=route_entry_id,
                                    batch_size=config.get('apply', 16),
                                    render_map=render_map)



