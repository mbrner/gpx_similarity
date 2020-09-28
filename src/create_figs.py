import csv
import pathlib
import datetime
import functools
import io

import click
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

import gpxpy
from gpxpy.geo import simplify_polyline
from gpxpy.gpx import GPXTrackPoint

import geotiler
from geotiler.cache import redis_downloader
import redis

from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker


from .model import get_engine_and_model


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
                     df.loc[i-1, 'distance_total'] + dist if prev_point is not None else 0.,
                    ]
        prev_point = p
    return df


def create_imgages(dbsession, model, gpx_file, show_route=True, route_type='unknown', dataset='unknown', zoom=16, size=(256, 256), max_distance=5, render_map=geotiler.render_map, dpi=100):
    width, height = size
    origin = str(gpx_file)
    delete_q = model.__table__.delete().where(model.origin==origin and
                                              model.show_route==show_route and
                                              model.zoom==zoom and
                                              model.width == width and
                                              model.height==height)
    dbsession.execute(delete_q)
    dbsession.commit()
    try:
        gpx = gpxpy.parse(gpx_file.open())
    except gpxpy.gpx.GPXXMLSyntaxException:
        print(f'{gpx_file} is not a valid GPX file!')
        return
    for track_idx, track in enumerate(gpx.tracks):
        for segment_idx, segment in enumerate(track.segments):
            _create_images_single_segment(
                dbsession,
                model,
                origin,
                segment,
                track_idx,
                segment_idx,
                show_route,
                route_type,
                dataset,
                zoom,
                size,
                max_distance,
                render_map,
                dpi)


def _create_images_single_segment(dbsession, model, path, segment, track_idx, segment_idx, show_route=True, route_type='unknown', dataset='unknown', zoom=16, size=(256, 256), max_distance=5, render_map=geotiler.render_map, dpi=100):
    width, height = size
    raw_points = segment.points
    points = simplify_polyline(raw_points, max_distance=5)
    df = df_from_points(points)
    if raw_points is not None:
        raw_points_lat, raw_points_long = points2array(raw_points)
    f_lat, f_long = interp1d(df.distance_total, df.latitude), interp1d(df.distance_total, df.longitude)
    mm_zero = geotiler.Map(center=(points[0].longitude, points[0].latitude), zoom=zoom, size=size)
    p_0_long, p_0_lat, p_1_long, p_1_lat = mm_zero.extent
    distance = GPXTrackPoint(latitude=p_0_lat, longitude=p_1_long).distance_2d(GPXTrackPoint(latitude=p_1_lat, longitude=p_0_long))
    steps = np.arange(df.distance_total.values[0], df.distance_total.values[-1], distance / (2.*1.41421356237))
    new_entries = []
    for i, d in enumerate(steps):
        p_lat = f_lat(d)
        p_long = f_long(d)
        mm = geotiler.Map(center=(p_long, p_lat), zoom=zoom, size=size)
        img = render_map(mm)
        fig = plt.figure(figsize=(width/dpi, height/dpi))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        p_0_long, p_0_lat, p_1_long, p_1_lat = mm.extent
        if raw_points is not None and show_route:
            p_0_long, p_0_lat, p_1_long, p_1_lat = mm.extent
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
        new_entry = model(origin=str(path),
                          track_id=track_idx,
                          segment_id=segment_idx,
                          zoom=zoom,
                          idx=i,
                          p_0_lat=p_0_lat,
                          p_0_long=p_0_long,
                          p_1_lat=p_1_lat,
                          p_1_long=p_1_long,
                          width=width,
                          height=height,
                          show_route=show_route,
                          route_type=route_type,
                          dataset=dataset,
                          image=buf.read())
        new_entries.append(new_entry)
        plt.close(fig)
    dbsession.add_all(new_entries)
    dbsession.commit()


def shrink_fov(p_low, p_high, factor=0.95):
    mid = (p_low + p_high) / 2.
    offset = (p_high - p_low) / 2.
    return mid - (offset * factor), mid + (offset * factor)
    

def add_train_files(config, in_files, dataset_name, default_route_type, extract_route_type, expand_paths):

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
            create_imgages(session,
                           OSMImages,
                           path,
                           route_type=route_type,
                           render_map=render_map,
                           show_route=bool(opts['show_route']),
                           zoom=opts['zoom'],
                           size=(opts['width'], opts['width']),
                           max_distance=opts['smoothing_dist'],
                           dataset=dataset_name)