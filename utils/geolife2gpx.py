"""Script use to convert the geolife1.3 dataset (download: https://www.microsoft.com/en-us/download/details.aspx?id=52367) to gpx files"""
import pathlib
import datetime
import csv


import numpy as np
import gpxpy
import click


class Labels:
    def __init__(self, start=[], stop=[], labels=[]):
        self.start = np.asarray(start)
        self.stop = np.asarray(stop)
        self.labels = labels
        
    def __call__(self, time):
        if len(self.start) == 0:
            return None
        if isinstance(time, datetime.datetime):
            time = time.timestamp()
        elif isinstance(time, float):
            pass
        else:
            raise AttributeError('time hast to be either datetime.datetime or float')
        idx = np.where(np.logical_and(self.start <= time, self.stop > time))[0]
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            return self.labels[0]
        else:
            return None


def parse_label_txt(label_txt, skip_lines=1, fmt_str='%Y/%m/%d %H:%S:%M'):
    start = []
    stop = []
    idx = []
    with label_txt.open() as stream:
        csv_reader = csv.reader(stream, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i < skip_lines:
                continue
            start.append(datetime.datetime.strptime(row[0], fmt_str).timestamp())
            stop.append(datetime.datetime.strptime(row[1], fmt_str).timestamp())
            idx.append(row[2])
    return Labels(start, stop, idx)


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


def parse_plt_dir(in_dir, out_dir):
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpx_files = []
    label_file = in_dir / 'labels.txt'
    if label_file.exists():
        labels = parse_label_txt(label_file)
    else:
        labels = lambda x: None
    traj_dir = in_dir / 'Trajectory'
    if not traj_dir.exists():
        raise ValueError(f'{traj_dir} is no geolife direcotry!')
    for plt_file in traj_dir.glob('*.plt'):
        gpx = gpx_from_plt(plt_file)
        label = labels(gpx.tracks[0].segments[0].points[0].time)
        name = f'{in_dir.stem}_{plt_file.stem}_{"unknown" if label is None else label}'
        gpx.tracks[0].name = name
        gpx_file = out_dir / f'{name}.gpx'
        with gpx_file.open('w') as stream:
            stream.write(gpx.to_xml())
        gpx_files.append(gpx_file)
    return gpx_files    
    

@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('out_dir')
def main(in_dir, out_dir):
    """This scripts transforms the unpacked geolife 1.3
    dataset (https://www.microsoft.com/en-us/download/details.aspx?id=52367) to gpx files.

    IN_DIR should be the `Geolife Trajectories 1.3/Data` folder.
    OUT_DIR is path where the gpx will be stored.
    """
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)
    for p in in_dir.glob('*'):
        if p.is_dir():
            try:
                parse_plt_dir(p, out_dir)
            except ValueError:
                click.echo(f'Directory {p} was skipped because it had not the expected folders!')


if __name__ == '__main__':
    main()