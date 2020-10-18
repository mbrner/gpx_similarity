import pathlib
import functools
import json
import copy

import numpy as np
import tensorflow as tf
import click
import geotiler
from geotiler.cache import redis_downloader
import gpxpy
from gpxpy.gpx import GPXTrackPoint
from gpxpy.geo import simplify_polyline
import redis
from sqlalchemy.orm import sessionmaker
from scipy.stats import trim_mean
from scipy.spatial.distance import cdist, pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from .create_figs import (zoom_map,
                          apply_model,
                          get_nn_model,
                          generate_images_for_segment,
                          bytes2array,
                          decompress_gpx,
                          df_from_points)
from .model import get_engine_and_model

HERE = pathlib.Path(__file__).parent
STYLES = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

ASSESTS_PATH = HERE / 'dash_assets'

COLOR_UNMARKED = '#503047'
COLOR_MARKED = '#C05746'

SIZE_NORAML = 15
COLOR_NOMATCH = '#7B68EE'
COLOR_MATCH = '#32CD32'

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return copy.deepcopy(d)


def prepare_df_for_plotting(df):
    df['Segment'] = df.index.values + 1
    return df.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})

def generate_segements_sim(config, session, Segments, Routes, segment_ids, route_ids, matrix):
    map_options = config['map_options']
    ref_routes = np.unique(route_ids)
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
    
    mean_dist[mean_dist < 0.006] = np.inf
    idx_min = np.argmin(mean_dist)
    path_most_sim = ref_files[ref_routes[idx_min]]
    route_entry = session.query(Routes).filter(Routes.path == path_most_sim).first()
    route_mask = route_ids == ref_routes[idx_min]
    matrix_sim = matrix[:, route_mask]
    segments = []
    for idx in segment_ids[route_mask]:
        segment = session.query(Segments).filter(Segments.id == int(idx)).first()
        mm = geotiler.Map(extent=(segment.p_0_long, segment.p_0_lat, segment.p_1_long, segment.p_1_lat), zoom=map_options['zoom'])
        mm = geotiler.Map(center=mm.center, zoom=map_options['zoom'], size=(map_options['width'], map_options['width']))
        segments.append({'point': GPXTrackPoint(latitude=mm.center[1], longitude=mm.center[0]), 'origin': route_entry.path})
    return segments, matrix_sim, route_entry.gpx_file



def generate_img_sim_global(config, session, Segments, matrix, segment_ids, embedding_model, render_map):
    map_options = config['map_options']
    idx = np.argmin(matrix, axis=1)
    min_values = np.min(matrix, axis=1)
    segment_idx = segment_ids[idx]
    segments = []
    images = []
    entries = []
    batch_size = config.get('apply').get('batch_size', 16)
    for (idx, value) in zip(segment_idx, min_values):
        entry = session.query(Segments).filter(Segments.id == int(idx)).first()
        entry = row2dict(entry)
        mm = geotiler.Map(extent=(entry['p_0_long'], entry['p_0_lat'], entry['p_1_long'], entry['p_1_lat']), zoom=map_options['zoom'])
        mm = geotiler.Map(center=mm.center, zoom=map_options['zoom'], size=(map_options['width'], map_options['width']))
        entry['point'] = GPXTrackPoint(latitude=mm.center[1], longitude=mm.center[0])
        entry['image_raw'] = render_map(mm).convert('RGB')
        entry['image_encoded'] = tf.reshape(tf.convert_to_tensor(bytes2array(entry['image']), dtype=tf.float32), map_options['encoding_shape'])
        entry['image_for_reconstruction'] = tf.keras.preprocessing.image.img_to_array(entry['image_raw']) / 255.
        entry['dist'] = value
        del entry['image']
        entries.append(entry)
        images.append(entry['image_for_reconstruction'])
        if len(images) == batch_size:
            reconstructed_images, *_ = apply_model(embedding_model, images, batch_size, fill_up=False)
            for img_decoded, entry in zip(reconstructed_images, entries):
                entry['image_decoded'] = img_decoded.numpy()
                segments.append(entry)
            images, entries = [], []
    if len(entries) > 0:
        reconstructed_images, *_ = apply_model(embedding_model, images, batch_size, fill_up=True)
        for img_decoded, entry in zip(reconstructed_images, entries):
            entry['image_decoded'] = img_decoded.numpy()
            segments.append(entry)
    return segments


def create_plotly_fig_with_line(config, gpx):
    map_options = config['map_options']
    map_options['size'] = (map_options['width'], map_options['height'])
    raw_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            raw_points.extend(segment.points)
    raw_points = simplify_polyline(raw_points, max_distance=map_options['smoothing_dist'])
    df_raw = df_from_points(raw_points)
    center_lat = (df_raw.latitude.min() + df_raw.latitude.max()) / 2.
    center_long = (df_raw.longitude.min() + df_raw.longitude.max()) / 2.
    mm = geotiler.Map(center=(center_long, center_lat), zoom=map_options['zoom'], size=map_options['size'])
    mm = zoom_map(mm, df_raw.longitude, df_raw.latitude)
    fig = go.FigureWidget(go.Scattermapbox(
            lat=df_raw.latitude,
            lon=df_raw.longitude,
            hoverinfo="none",
            showlegend=False,
            mode='lines'))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=mm.center[1],
                lon=mm.center[0]),
            pitch=0,
            zoom=mm.zoom
        )
    )
    fig.update_traces(line=dict(color="Black", width=0.5))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def apply_model_to_file(config, gpx_file, ref_database, checkpoint):
    gpx_file = pathlib.Path(gpx_file)
    click.echo(f'Loading model from: {checkpoint}!')
    embedding_model = tf.keras.models.load_model(checkpoint)
    client = redis.Redis(**config['redis'])
    downloader = redis_downloader(client)
    render_map = functools.partial(geotiler.render_map, downloader=downloader)
    batch_size = config.get('apply').get('batch_size', 16)

    map_options = config['map_options']
    map_options['size'] = (map_options['width'], map_options['height'])
    click.echo(f'Generating segment images for the test gpx: {gpx_file}...')
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
                    reconstructed_images, mu, log_var = apply_model(embedding_model, images, batch_size, fill_up=False)
                    images = [np.array((mu_i, log_var_i)) for mu_i, log_var_i in zip(log_var, mu)]
                    for i, (img_encoded, img_decoded, entry) in enumerate(zip(images, reconstructed_images, entries)):
                        if i == 0:
                            map_options['encoding_shape'] = img_encoded.shape
                        entry['image_encoded'] = img_encoded
                        entry['image_decoded'] = img_decoded.numpy()
                        segments.append(entry)
                    images, entries = [], []
            if len(entries) > 0:
                reconstructed_images, mu, log_var = apply_model(embedding_model, images, batch_size, fill_up=False)
                images = [np.array((mu_i, log_var_i)) for mu_i, log_var_i in zip(log_var, mu)]
                for img_encoded, img_decoded, entry in zip(images, reconstructed_images, entries):
                    entry['image_encoded'] = img_encoded
                    entry['image_decoded'] = img_decoded.numpy()
                    segments.append(entry)
    test_images = np.asarray([entry['image_encoded'] for entry in segments])
    click.echo(f'Loading date from reference database: {ref_database}...')
    engine, Routes, Segments = get_engine_and_model(ref_database, train=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    images = []
    route_ids = []
    segment_ids = []
    for seg in session.query(Segments.image, Segments.origin, Segments.id):
        images.append(bytes2array(seg.image).reshape(map_options['encoding_shape']))
        route_ids.append(seg.origin)
        segment_ids.append(seg.id)
    images = np.asarray(images)
    route_ids = np.asarray(route_ids)
    segment_ids = np.asarray(segment_ids)    
    click.echo(f'Calculating Bhattacharyya distance...')

    def bhattacharyya_distance(vec_a, vec_b):
        mu_a, var_a = vec_a[:len(vec_a)//2], np.exp(vec_a[len(vec_a)//2:])
        mu_b, var_b = vec_b[:len(vec_b)//2], np.exp(vec_b[len(vec_b)//2:])
        result = 0.25 * np.log(0.25*(var_a/var_b+var_b/var_a+2)) + 0.25 * ((mu_a-mu_b)**2 / (var_a+var_b))
        return np.mean(result)


    test_images_reshaped = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])
    images_reshaped = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    matrix = cdist(test_images_reshaped, images_reshaped, metric=bhattacharyya_distance)
    click.echo(f'Determine best matching route via aggregation of segments with `{config["apply"]["aggregation"]}`...')
    segments_sim, matrix_sim, compressed_gpx_file = generate_segements_sim(config, session, Segments, Routes, segment_ids, route_ids, matrix)
    click.echo(f'Preparing images of similar segments...')
    segments_sim_global = generate_img_sim_global(config, session, Segments, matrix, segment_ids, embedding_model, render_map)

    fig_test = create_plotly_fig_with_line(config, gpx)
    fig_sim = create_plotly_fig_with_line(config, gpxpy.parse(decompress_gpx(compressed_gpx_file)))
    session.close()
    return fig_test, fig_sim, segments, segments_sim, segments_sim_global, matrix_sim


def run_comparison(config, gpx_file, ref_database, checkpoint):
    fig_test, fig_sim, segments_test, segments_sim, segments_sim_global, matrix_sim = apply_model_to_file(config, gpx_file, ref_database, checkpoint)

    log_matrix = np.log2(1. / matrix_sim)
    mean, std = np.mean(log_matrix), np.std(log_matrix)

    def normalize_marker(values, min_scale=2., max_scale=20.):
        values = np.log2(1. / values)
        normalized = (values - mean) / std
        sig = 1/(1 + np.exp(-normalized))
        marker_scaling = sig
        min_scale = 2.
        max_scale = 20.
        return min_scale + (max_scale - min_scale) * marker_scaling

    df_test = df_from_points([entry['point'] for entry in segments_test])
    df_test = prepare_df_for_plotting(df_test)
    scatter_test = px.scatter_mapbox(df_test, lat="Latitude", lon="Longitude", hover_data=["Segment"]).data[0]
    n_markers_test = len(segments_test)
    colors = [COLOR_UNMARKED] * n_markers_test
    scatter_test.marker.color = colors
    fig_test.add_trace(scatter_test)
    fig_test.update_traces(marker_size=SIZE_NORAML)

    df_sim = df_from_points([entry['point'] for entry in segments_sim])
    df_sim = prepare_df_for_plotting(df_sim)
    scatter_sim = px.scatter_mapbox(df_sim, lat="Latitude", lon="Longitude", hover_data=["Segment"]).data[0]
    n_markers_sim = len(segments_sim)
    colors = [COLOR_NOMATCH] * n_markers_sim
    scatter_sim.marker.color = colors
    fig_sim.add_trace(scatter_sim)
    fig_sim.update_traces(marker_size=SIZE_NORAML)

    fig_seg_test = go.Figure(
        px.imshow(np.zeros((config['map_options']['width'], config['map_options']['height'], 3)))
    )
    fig_seg_sim = go.Figure(
        px.imshow(np.zeros((config['map_options']['width'], config['map_options']['height'], 3)))
    )
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash('GPX Similarity', assets_folder=ASSESTS_PATH)
    app.layout = html.Div([
        dcc.Store(id='clicked-data'),
        html.H2(f'Loaded File: {gpx_file}'),
        html.Div([dcc.Graph(figure=fig_test, id='main-map'),]),

        html.Div([html.Div([html.H5('Selected Segment in Test Route'),
                            dcc.Graph(id='fig_seg_ref', figure=fig_seg_test)],
                           className="six columns"),
                  html.Div([html.H5('Most Similar Segment in Database'),
                            dcc.Graph(id='fig_seg_sim', figure=fig_seg_sim)],
                           className="six columns"),],
                className="row"),
        dcc.Checklist(id='checkbox', options=[{'label': 'Show embedded', 'value': 'embedded'}], value=[]),
        html.H2(f'Most Similar Route: {segments_sim[0]["origin"]}'),
        html.Div([dcc.Graph(figure=fig_sim, id='sim-map'),]),
        html.Div([dcc.Markdown((HERE / 'dash_assets' / 'comparison_explanation.md').open().read())], className='three columns')])

    @app.callback([Output('fig_seg_ref', 'figure'),
                   Output('fig_seg_sim', 'figure'),
                   Output('main-map', 'figure'),
                   Output('sim-map', 'figure')],
                  [Input('main-map', 'clickData'),
                   Input('checkbox', 'value')],
                  [State('main-map', 'figure'),
                   State('sim-map', 'figure')])
    def display_click_data(click_data, checkbox_value, figure_test, figure_sim):
        figure_test = go.FigureWidget(figure_test)
        figure_sim = go.FigureWidget(figure_sim)
        try:
            idx = click_data['points'][0]['customdata'][0]-1
        except (KeyError, IndexError, TypeError):
            img_test = np.zeros((config['map_options']['width'], config['map_options']['height'], 3))
            img_sim = np.zeros((config['map_options']['width'], config['map_options']['height'], 3))

            colors = [COLOR_UNMARKED] * n_markers_test
            figure_test.data[1].marker.color = colors

            size = [SIZE_NORAML] * n_markers_sim
            figure_sim.data[1].marker.size = size
        else:
            if 'embedded' in checkbox_value:
                key = 'image_decoded'
            else:
                key = 'image_raw'
            entry_test = segments_test[idx]
            entry_sim_global = segments_sim_global[idx]
            img_test = entry_test[key]
            img_sim = entry_sim_global[key]

            colors = [COLOR_UNMARKED] * n_markers_test
            colors[click_data['points'][0]['pointIndex']] = COLOR_MARKED
            figure_test.data[1].marker.color = colors

            size = normalize_marker(matrix_sim[idx, :])
            figure_sim.data[1].marker.size = size

        fig_seg_test = go.Figure(px.imshow(img_test))
        fig_seg_sim = go.Figure(px.imshow(img_sim))
        return fig_seg_test, fig_seg_sim, figure_test, figure_sim

    click.echo(f'Starting Dash+plotly visualization...')
    app.title = 'GPX Similarity'
    app.run_server(debug=True, use_reloader=False) 