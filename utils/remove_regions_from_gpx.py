"""Small to script to remove certain areas from a given gpx file"""
import pathlib

import click
import gpxpy
from gpxpy.gpx import GPXTrackPoint
from typing import List, Tuple


def clean_gpx(in_file: pathlib.Path, out_file: pathlib.Path, forbidden_regions: List[Tuple[GPXTrackPoint, float]]=[]):
    gpx_file = gpxpy.parse(in_file.open())
    cleaned_tracks = []
    for track in gpx_file.tracks:
        cleaned_segments = []
        for segment in track.segments:
            cleaned_points = []
            for point in segment.points:
                remove = False
                for (center, radius) in forbidden_regions:
                    distance = center.distance_2d(point)
                    if distance < radius:
                        remove = True
                        break
                if not remove:
                    cleaned_points.append(point)
            if len(cleaned_points) > 0:
                segment.points = cleaned_points
                cleaned_segments.append(segment)
        if len(cleaned_segments) > 0:
            track.segments = cleaned_segments
            cleaned_tracks.append(track)
    if len(cleaned_tracks) > 0:
        gpx_file.tracks = cleaned_tracks
        with out_file.open('w') as stream:
            stream.write(gpx_file.to_xml())
    else:
        raise ValueError('No points left after cleaning! No new gpx file written!')


@click.command()  # @cli, not @click!
@click.argument('in_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--forbidden_regions', '-r', type=(float, float, float), multiple=True, required=True)
def main(in_file, out_file, forbidden_regions):
    in_file = pathlib.Path(in_file)
    out_file = pathlib.Path(out_file)
    forbidden_regions_parsed = []
    for (p_lat, p_long, radius) in forbidden_regions:
        forbidden_regions_parsed.append((GPXTrackPoint(latitude=p_lat, longitude=p_long), radius))
    clean_gpx(in_file, out_file, forbidden_regions=forbidden_regions_parsed)


if __name__ == '__main__':
    main()