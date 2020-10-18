import click
import toml
import pathlib


HERE = pathlib.Path(__file__).parent



@click.group()
def cli():
    pass

cli.__doc__ = (HERE / 'README.md').open().read().partition('<!--- HIDE IN CLICK --->')[0]  # The help string in the click CLI is taken from the README.
                                                                                 # Everything in the README before `< !-- HIDE IN CLICK --/>`
                                                                                 # is used.
cli.help =  cli.__doc__

@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('in_files', nargs=-1)
@click.option('--type-extraction/--no-type-extraction', 'type_extraction', default=False)
@click.option('--absolute-paths/--relativ-paths', 'expand_paths', default=False)
@click.option('--skip_existing/--replace_existing', 'skip_existing', default=False)
@click.option('-d', '--dataset-name', 'dataset_name', default='unknown')
@click.option('-r', '--route-type', 'route_type', default='unknown')
def add_train_files(type_extraction, config, dataset_name, route_type, in_files, expand_paths, skip_existing):
    """Add .gpx files to the training database.
    Add IN_FILES (.gpx files) as images of segements of the route to the train database.
    The settings for the process are taken from CONFIG.
    BEWARE: postgres and redis have to be running for all training steps."""
    from source.create_figs import add_train_files
    if len(in_files) > 3:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}',... '{in_files[-1]}') [{len(in_files)} files]"
    else:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}', '{in_files[2]}') [{len(in_files)} files]"
    msg = [f'Adding files: {in_files_str} to train dataset: "{dataset_name}"']
    msg.append(f'\tconfig: {config}')
    msg.append(f'\textract route_type: {type_extraction} [default="{route_type}"]')
    click.echo('\n'.join(msg))
    config = toml.load(config)
    add_train_files(config=config,
                    in_files=in_files,
                    dataset_name=dataset_name,
                    default_route_type=route_type,
                    extract_route_type=type_extraction,
                    expand_paths=expand_paths,
                    skip_existing=skip_existing)


@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('-c', '--checkpoint', 'checkpoint',
              default=None,
              type=click.Path(exists=True),
              help='Initial chcpoint for the model. If None are provided the '
                   'model weights are initialized random.')
def train_model(config, output_dir, checkpoint):
    """Train the model.
    The options for the training are taken from CONFIG.
    Training infos and model weights are saved in the OUTPUT_DIR"""
    from source.nn_train import train
    config = toml.load(config)
    train(config, output_dir, checkpoint)


@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('reference_database', type=click.Path())
@click.argument('checkpoint', type=click.Path())
@click.argument('in_files', nargs=-1)
@click.option('--type-extraction/--no-type-extraction', 'type_extraction', default=False)
@click.option('--absolute-paths/--relativ-paths', 'expand_paths', default=False)
@click.option('--skip_existing/--replace_existing', 'skip_existing', default=False)
@click.option('-d', '--dataset-name', 'dataset_name', default='unknown')
@click.option('-r', '--route-type', 'route_type', default='unknown')
def add_reference_files(type_extraction, config, dataset_name, checkpoint, route_type, in_files, reference_database, expand_paths, skip_existing):
    """Add .gpx files to the reference database.
    Add IN_FILES (.gpx files) as images of segements of the route to the REFERENCE_DATABASE.
    The settings for the process are taken from CONFIG.
    The segemnts are added to the database after encoding with the model provided via training CHECKPOINT.
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from source.create_figs import add_reference_files
    if len(in_files) > 3:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}',... '{in_files[-1]}') [{len(in_files)} files]"
    else:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}', '{in_files[2]}') [{len(in_files)} files]"
    msg = [f'Adding reference files: {in_files_str} to dataset "{dataset_name}" in database {reference_database}']
    msg.append(f'\tconfig: {config}')
    msg.append(f'\textract route_type: {type_extraction} [default="{route_type}"]')
    click.echo('\n'.join(msg))
    config = toml.load(config)
    add_reference_files(config=config,
                        checkpoint=checkpoint,
                        reference_database=reference_database,
                        in_files=in_files,
                        dataset_name=dataset_name,
                        default_route_type=route_type,
                        extract_route_type=type_extraction,
                        expand_paths=expand_paths,
                        skip_existing=skip_existing)



@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('reference_database', type=click.Path(exists=True))
@click.argument('checkpoint', type=click.Path())
@click.argument('in_file', type=click.Path(exists=True))
def compare_gpx(config, in_file, reference_database, checkpoint):
    """Run a comparison of a gpx-file against a reference database and show result in the browser.
    Compare a gpx file against a REFERENCE_DATABASE.
    The provided IN_FILE (.gpx file) is compared segment-wise against the REFERENCE_DATABASE.
    The comparison in done via calculating the `Bhattacharyya` distance in the latent space.
    For reasonable result us the CHECKPOINT also used to create the REFERENCE_DATABASE.
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from source.compare import run_comparison
    #tf.logging.set_verbosity(tf.logging.ERROR)
    click.echo(f'Loading config: {config}')
    config = toml.load(config)
    run_comparison(config, in_file, reference_database, checkpoint)

if __name__ == '__main__':
    cli()



