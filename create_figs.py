import csv
import pathlib
import datetime
import functools
import io

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



def gpx_from_plt(plt_path, name=None, skip_lines=6, fmt_str='%Y-%m-%d %H:%S:%M'):
    gpx = gpxpy.gpx.GPX()

    gpx_track = gpxpy.gpx.GPXTrack(name=name)
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)


    skip_lines = 6
    with plt_path.open() as stream:
        csv_reader = csv.reader(stream)
        for i, row in enumerate(csv_reader):
            if i < skip_lines:
                continue
            time = datetime.datetime.strptime(f'{row[5]} {row[6]}', fmt_str)
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=float(row[0]),
                                                              longitude=float(row[1]),
                                                              elevation=float(row[3])*0.3048,
                                                              time=time))
    return gpx

def create_imgages(dbsession, model, gpx_file, show_route=True, route_type='unknown', zoom=16, size=(256, 256), max_distance=5, render_map=geotiler.render_map):
    origin=str(gpx_file)
    delete_q = model.__table__.delete().where(model.origin==origin and model.show_route==show_route and model.zoom==zoom)
    dbsession.execute(delete_q)
    dbsession.commit()
    gpx = gpxpy.parse(gpx_file.open())
    raw_points = gpx.tracks[0].segments[0].points
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
        fig = plt.figure(figsize=(10, 10))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        p_0_long, p_0_lat, p_1_long, p_1_lat = mm.extent
        if raw_points is not None and show_route:
            p_0_long, p_0_lat, p_1_long, p_1_lat = mm.extent
            p_0_long_shrunk, p_1_long_shrunk = shrink_fov(p_0_long, p_1_long)
            p_0_lat_shrunk, p_1_lat_shrunk = shrink_fov(p_0_lat, p_1_lat)
            idx_lat = np.logical_and(p_0_lat_shrunk <= raw_points_lat, p_1_lat_shrunk >= raw_points_lat)
            idx_long = np.logical_and(p_0_long_shrunk <= raw_points_long, p_1_long_shrunk >= raw_points_long)
            idx = np.logical_and(idx_lat, idx_long)
            selected_lat, selected_long = raw_points_lat[idx], raw_points_long[idx]
            track = np.array([mm.rev_geocode(p) for p in zip(selected_long, selected_lat)])
            if len(track.shape) > 1:
                ax.plot(track[:, 0], track[:, 1], 'r-', alpha=1.0, lw=10, ms=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        new_entry = model(origin=str(gpx_file),
                          zoom=zoom,
                          idx=i,
                          p_0_lat=p_0_lat,
                          p_0_long=p_0_long,
                          p_1_lat=p_1_lat,
                          p_1_long=p_1_long,
                          show_route=show_route,
                          route_type=route_type,
                          image=buf.read())
        new_entries.append(new_entry)
        plt.close(fig)
    dbsession.add_all(new_entries)
    dbsession.commit()

def shrink_fov(p_low, p_high, factor=0.95):
    mid = (p_low + p_high) / 2.
    offset = (p_high - p_low) / 2.
    return mid - (offset * factor), mid + (offset * factor)
    
    
if __name__ == '__main__':
    from model import OSMImages, ENGINE
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=ENGINE)
    session = Session()
    client = redis.Redis('localhost')
    downloader = redis_downloader(client)
    render_map = functools.partial(geotiler.render_map, downloader=downloader)
    in_dir = pathlib.Path('/home/mathis/Projects/gpx_similarity/geo_life_1.3_gpx')
    for i, p in enumerate(in_dir.glob('*bike.gpx')):
        print(p)
        create_imgages(session, OSMImages, p, route_type='bike', show_route=False, render_map=render_map)